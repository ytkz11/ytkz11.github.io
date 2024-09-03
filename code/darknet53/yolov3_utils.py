class YoloLayer(nn.Module):
    def __init__(self, anchors, num_classes, img_dim=416, ignore_thres=0.5):
        super(YoloLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = ignore_thres
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCELoss(reduction='sum')
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0  # grid size

    def build_targets(self, pred_boxes, pred_cls, target, anchors, ignore_thres):
        nB = pred_boxes.size(0)
        nA = pred_boxes.size(1)
        nC = pred_cls.size(-1)
        nG = pred_boxes.size(2)

        # Output tensors
        obj_mask = pred_boxes.new(nB, nA, nG, nG).fill_(0)
        noobj_mask = pred_boxes.new(nB, nA, nG, nG).fill_(1)
        class_mask = pred_boxes.new(nB, nA, nG, nG).byte()
        tx = pred_boxes.new(nB, nA, nG, nG).zero_()
        ty = pred_boxes.new(nB, nA, nG, nG).zero_()
        tw = pred_boxes.new(nB, nA, nG, nG).zero_()
        th = pred_boxes.new(nB, nA, nG, nG).zero_()
        tconf = pred_boxes.new(nB, nA, nG, nG).zero_()
        tcls = pred_boxes.new(nB, nA, nG, nG, nC).zero_()

        # Convert to position relative to box
        target_boxes = target[:, 2:6] * nG
        gxy = target_boxes[:, :2]
        gwh = target_boxes[:, 2:]
        ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])

        best_ious, best_n = ious.max(0)
        b, target_labels = target[:, :2].long().t()

        gx, gy = gxy.t()
        gw, gh = gwh.t()
        gi, gj = gxy.long().t()

        # Set masks
        obj_mask[b, best_n, gj, gi] = 1
        noobj_mask[b, best_n, gj, gi] = 0

        # Set noobj mask to zero where iou exceeds ignore threshold
        for i, anchor_ious in enumerate(ious.t()):
            noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

        # Coordinates
        tx[b, best_n, gj, gi] = gx - gx.floor()
        ty[b, best_n, gj, gi] = gy - gy.floor()
        # Width and height
        tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
        th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
        # One-hot encoding of label
        tcls[b, best_n, gj, gi, target_labels] = 1
        # Compute label correctness and iou at best anchor
        class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()

        tconf[b, best_n, gj, gi] = obj_mask[b, best_n, gj, gi].float()

        return obj_mask, noobj_mask, class_mask, tx, ty, tw, th, tconf, tcls

    def forward(self, output, targets=None):
        nB = output.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        nG = output.size(2)

        # Get outputs
        x = torch.sigmoid(output[..., 0])  # Center x
        y = torch.sigmoid(output[..., 1])  # Center y
        w = output[..., 2]  # Width
        h = output[..., 3]  # Height
        conf = torch.sigmoid(output[..., 4])  # Conf
        pred_cls = torch.sigmoid(output[..., 5:])  # Cls pred.

        # Calculate offsets for each grid
        grid_x = torch.arange(nG).repeat(nG, 1).view([1, 1, nG, nG]).type_as(output)
        grid_y = torch.arange(nG).repeat(nG, 1).t().view([1, 1, nG, nG]).type_as(output)
        scaled_anchors = torch.FloatTensor([(a_w / self.img_dim, a_h / self.img_dim) for a_w, a_h in self.anchors]).type_as(output)
        anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))

        pred_boxes = torch.zeros((nB, nA, nG, nG, 4)).type_as(output)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        if targets is not None:
            obj_mask, noobj_mask, class_mask, tx, ty, tw, th, tconf, tcls = self.build_targets(pred_boxes=pred_boxes,
                                                                                               pred_cls=pred_cls,
                                                                                               target=targets,
                                                                                               anchors=scaled_anchors,
                                                                                               ignore_thres=self.ignore_thres)

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_conf_obj = self.bce_loss(conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = self.bce_loss(pred_cls[class_mask], tcls[class_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics
            cls_acc = 100 * class_mask.float().mean()
            conf_obj = conf[obj_mask].mean()
            conf_noobj = conf[noobj_mask].mean()
            conf50 = (conf > 0.5).float()
            iou50 = (ious.view(-1) > 0.5).float()
            iou75 = (ious.view(-1) > 0.75).float()
            detected_mask = conf50 * class_mask

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(detected_mask * iou50).mean().item(),
                "recall75": to_cpu(detected_mask * iou75).mean().item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": nG
            }

            return total_loss
        else:
            return output, pred_boxes
        

def load_classes(path):
    """
    Loads class labels at 'path'
    """
    with open(path, 'r') as file:
        names = file.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)
