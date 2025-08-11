import os

def find_string_in_files(folder_path, search_string):
    """
    在指定文件夹中搜索包含特定字符串的文件
    Args:
        folder_path (str): 要搜索的文件夹路径
        search_string (str): 要查找的字符串
    """
    try:
        # 检查文件夹路径是否有效
        if not os.path.isdir(folder_path):
            print(f"错误：'{folder_path}' 不是一个有效的文件夹路径")
            return

        # 遍历文件夹中的所有文件
        for root, _, files in os.walk(folder_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                
                # 仅处理文本文件（可以根据需要扩展文件类型）
                if file_name.endswith(('.txt', '.py', '.csv', '.log', '.md')):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as file:
                            content = file.read()
                            if search_string in content:
                                print(f"找到匹配文件: {file_path}")
                    except (UnicodeDecodeError, IOError) as e:
                        print(f"无法读取文件 {file_path}: {e}")

    except Exception as e:
        print(f"发生错误: {e}")

def main():
    # 获取用户输入
    folder_path = input("请输入文件夹路径: ")
    search_string = input("请输入要查找的字符串: ")
    
    print(f"\n正在搜索包含 '{search_string}' 的文件...")
    find_string_in_files(folder_path, search_string)
    print("搜索完成！")


if __name__ == "__main__":
    main()
    #hidden: ture