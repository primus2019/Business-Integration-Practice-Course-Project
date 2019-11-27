import os


def printlog(content, file_path='logs/default.log', linebreak=True, encoding='utf-8', creative=False, printable=True):
    if printable:
        print(content)
    if creative:
        if not os.path.isdir(os.path.dirname(file_path)):
            os.mkdir(os.path.dirname(file_path))
        if not os.path.isfile(file_path):
            open(file_path, 'a+').close()
    assert os.path.isdir(os.path.dirname(file_path)), 'Log.log: directory {} does not exist'.format(file_path)
    assert os.path.isfile(file_path), 'Log.log: file {} does not exist'.format(os.path.basename(file_path))
    with open(file_path, 'a') as file:
        if linebreak:
            file.write((str)(content) + '\n')
        else:
            file.write((str)(content))


def log(content, file_path='logs/default.log', linebreak=True, encoding='utf-8', creative=False, printable=False):
    if printable:
        print(content)
    if creative:
        if not os.path.isdir(os.path.dirname(file_path)):
            os.mkdir(os.path.dirname(file_path))
        if not os.path.isfile(file_path):
            open(file_path, 'a+').close()
    assert os.path.isdir(os.path.dirname(file_path)), 'Log.log: directory {} does not exist'.format(file_path)
    assert os.path.isfile(file_path), 'Log.log: file {} does not exist'.format(os.path.basename(file_path))
    with open(file_path, 'a') as file:
        if linebreak:
            file.write((str)(content) + '\n')
        else:
            file.write((str)(content))


def clear_log(file_path='logs/default.log', encoding='utf-8', creative=False):
    if creative:
        if not os.path.isdir(os.path.dirname(file_path)):
            os.mkdir(os.path.dirname(file_path))
        if not os.path.isfile(file_path):
            open(file_path, 'a+').close()
    assert os.path.isdir(os.path.dirname(file_path)), 'Log.clear_log: directory {} does not exist'.format(file_path)
    assert os.path.isfile(file_path), 'Log.clear_log: file {} does not exist'.format(os.path.basename(file_path))
    open(file_path, 'w').close()

def itersave(file_path, iteritem, encoding='utf-8'):
    with open(file_path, 'w+') as file:
        for item in iteritem:
            file.write('{}\n'.format(item))