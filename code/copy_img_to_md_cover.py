

import os
import re

from win32trace import flush

if __name__ == '__main__':
    path = r'D:\dengkaiyuan\code\my_blog\source\_postsCopy'

    # 获取path 下的后缀为md 的文件
    file_list = os.listdir(path)
    file_nohttp_list = []
    md_list = [os.path.join(path, x) for x in file_list if x.endswith('.md')]


    for j, md_file in enumerate(md_list):
        # md_file = md_list[56]
        with open(md_file, 'r', encoding='utf-8') as f:
            text = f.readlines()
            https_list = []
        cover_stytle = 0
        for i, t in enumerate(text[:10]):
            if 'cover: h' in t:
                cover_stytle = 1
            if t == 'cover: \n':
                del text[i]
        newtext = text
        if cover_stytle == 0:
            for t in text:
                if 'http' in t:

                    m = re.findall(r'\(http.+\)', t)
                    if m != []:
                        https_list.append(m[-1][1:-1])
                        print(https_list[-1])
       

        if https_list ==[]:
            file_nohttp_list.append(md_file)
        else:
            newtext.insert(3, 'cover: {}\n'.format(https_list[-1]))
            del newtext[4]
            os.makedirs(os.path.join(path, 'cover'), exist_ok=True)
            with open(os.path.join(os.path.join(path, 'cover'), os.path.basename(md_file)), 'w', encoding='utf-8') as f:
                f.writelines(newtext)
        print('\r{}%'.format(j / len(md_list) * 100), end="", flush=True)
    print()


