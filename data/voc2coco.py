import os
import random
import shutil
import json
import glob
import xml.etree.ElementTree as ET


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise ValueError("Can not find %s in %s." % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise ValueError(
            "The size of %s is supposed to be %d, but is %d."
            % (name, length, len(vars))
        )
    if length == 1:
        vars = vars[0]
    return vars


def get_filename_as_int(filename):
    try:
        filename = filename.replace("\\", "/")
        filename = os.path.splitext(os.path.basename(filename))[0]
        return int(filename)
    except:
        raise ValueError("Filename %s is supposed to be an integer." % (filename))


# 获取数据集中类别的名字
def get_categories(xml_files):
    classes_names = []
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            classes_names.append(member[0].text)
    classes_names = list(set(classes_names))
    classes_names.sort()
    print(f"类别名字为{classes_names}")
    return {name: i for i, name in enumerate(classes_names)}


def convert(xml_files, json_file):
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
    if PRE_DEFINE_CATEGORIES is not None:
        categories = PRE_DEFINE_CATEGORIES
    else:
        categories = get_categories(xml_files)
    bnd_id = START_BOUNDING_BOX_ID
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        path = get(root, "path")
        if len(path) == 1:
            filename = os.path.basename(path[0].text)
        elif len(path) == 0:
            filename = get_and_check(root, "filename", 1).text
        else:
            raise ValueError("%d paths found in %s" % (len(path), xml_file))
        ## The filename must be a number
        image_id = get_filename_as_int(filename)
        size = get_and_check(root, "size", 1)
        width = int(get_and_check(size, "width", 1).text)
        height = int(get_and_check(size, "height", 1).text)
        image = {
            "file_name": filename,
            "height": height,
            "width": width,
            "id": image_id,
        }
        json_dict["images"].append(image)
        ## Currently we do not support segmentation.
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'
        for obj in get(root, "object"):
            category = get_and_check(obj, "name", 1).text
            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, "bndbox", 1)
            xmin = int(get_and_check(bndbox, "xmin", 1).text) - 1
            ymin = int(float(get_and_check(bndbox, "ymin", 1).text)) - 1
            xmax = int(get_and_check(bndbox, "xmax", 1).text)
            ymax = int(get_and_check(bndbox, "ymax", 1).text)
            assert xmax > xmin
            assert ymax > ymin
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {
                "area": o_width * o_height,
                "iscrowd": 0,
                "image_id": image_id,
                "bbox": [xmin, ymin, o_width, o_height],
                "category_id": category_id,
                "id": bnd_id,
                "ignore": 0,
                "segmentation": [],
            }
            json_dict["annotations"].append(ann)
            bnd_id = bnd_id + 1

    for cate, cid in categories.items():
        cat = {"supercategory": "none", "id": cid, "name": cate}
        json_dict["categories"].append(cat)

    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    json_fp = open(json_file, "w")
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()


# 新建文件夹
def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' ----- folder created')
        return True
    else:
        print(path + ' ----- folder existed')
        return False


if __name__ == '__main__':
    # 验证集比例
    valRatio = 0.2
    # 测试集比例
    testRatio = 0
    # 获取当前脚本路径
    main_path = os.getcwd()
    # voc格式的图片和xml存放路径
    voc_images = os.path.join(main_path, 'VOC', 'JPEGImages')
    voc_annotations = os.path.join(main_path, 'VOC', 'Annotations')
    # 获取xml数量
    xmlNum = len(os.listdir(voc_annotations))

    val_files_num = int(xmlNum * valRatio)
    test_files_num = int(xmlNum * testRatio)

    coco_path = os.path.join(main_path, 'COCO')
    # coco_images = os.path.join(main_path, 'COCO', 'images')
    coco_json_annotations = os.path.join(main_path, 'COCO', 'annotations')
    coco_train2017 = os.path.join(main_path, 'COCO', 'train2017')
    coco_val2017 = os.path.join(main_path, 'COCO', 'val2017')
    coco_test2017 = os.path.join(main_path, 'COCO', 'test2017')
    xml_val = os.path.join(main_path, 'xml', 'xml_val')
    xml_test = os.path.join(main_path, 'xml', 'xml_test')
    xml_train = os.path.join(main_path, 'xml', 'xml_train')

    mkdir(coco_path)
    # mkdir(coco_images)
    mkdir(coco_json_annotations)
    mkdir(xml_val)
    mkdir(xml_test)
    mkdir(xml_train)
    mkdir(coco_train2017)
    mkdir(coco_val2017)
    if testRatio:
        mkdir(coco_test2017)

    for i in os.listdir(voc_images):
        img_path = os.path.join(voc_images, i)
        shutil.copy(img_path, coco_train2017)

        # voc images copy to coco images
    for i in os.listdir(voc_annotations):
        img_path = os.path.join(voc_annotations, i)
        shutil.copy(img_path, xml_train)

    print("\n\n %s files copied to %s" % (val_files_num, xml_val))

    for i in range(val_files_num):
        if len(os.listdir(xml_train)) > 0:

            random_file = random.choice(os.listdir(xml_train))
            #         print("%d) %s"%(i+1,random_file))
            source_file = "%s/%s" % (xml_train, random_file)
            # 分离文件名
            font, ext = random_file.split('.')
            valJpgPathList = [j for j in os.listdir(coco_train2017) if j.startswith(font)]
            if random_file not in os.listdir(xml_val):
                shutil.move(source_file, xml_val)
                shutil.move(os.path.join(coco_train2017, valJpgPathList[0]), coco_val2017)

            else:
                random_file = random.choice(os.listdir(xml_train))
                source_file = "%s/%s" % (xml_train, random_file)
                shutil.move(source_file, xml_val)
                # 分离文件名
                font, ext = random_file.split('.')
                valJpgPathList = [j for j in os.listdir(coco_train2017) if j.startswith(font)]
                shutil.move(os.path.join(coco_train2017, valJpgPathList[0]), coco_val2017)
        else:
            print('The folders are empty, please make sure there are enough %d file to move' % (val_files_num))
            break

    for i in range(test_files_num):
        if len(os.listdir(xml_train)) > 0:

            random_file = random.choice(os.listdir(xml_train))
            #         print("%d) %s"%(i+1,random_file))
            source_file = "%s/%s" % (xml_train, random_file)
            # 分离文件名
            font, ext = random_file.split('.')
            testJpgPathList = [j for j in os.listdir(coco_train2017) if j.startswith(font)]
            if random_file not in os.listdir(xml_test):
                shutil.move(source_file, xml_test)
                shutil.move(os.path.join(coco_train2017, testJpgPathList[0]), coco_test2017)
            else:
                random_file = random.choice(os.listdir(xml_train))
                source_file = "%s/%s" % (xml_train, random_file)
                shutil.move(source_file, xml_test)
                # 分离文件名
                font, ext = random_file.split('.')
                testJpgPathList = [j for j in os.listdir(coco_train2017) if j.startswith(font)]
                shutil.move(os.path.join(coco_train2017, testJpgPathList[0]), coco_test2017)
        else:
            print('The folders are empty, please make sure there are enough %d file to move' % (val_files_num))
            break

    print("\n\n" + "*" * 27 + "[ Done ! Go check your file ]" + "*" * 28)

    START_BOUNDING_BOX_ID = 1
    PRE_DEFINE_CATEGORIES = None

    xml_val_files = glob.glob(os.path.join(xml_val, "*.xml"))
    xml_test_files = glob.glob(os.path.join(xml_test, "*.xml"))
    xml_train_files = glob.glob(os.path.join(xml_train, "*.xml"))

    convert(xml_val_files, os.path.join(coco_json_annotations, 'val2017.json'))

    convert(xml_train_files, os.path.join(coco_json_annotations, 'train2017.json'))
    if testRatio:
        convert(xml_test_files, os.path.join(coco_json_annotations, 'test2017.json'))

    # 删除文件夹
    try:
        shutil.rmtree(xml_train)
        shutil.rmtree(xml_val)
        shutil.rmtree(xml_test)
        shutil.rmtree(os.path.join(main_path, 'xml'))
    except:
        print(f'xml文件删除失败，请手动删除{xml_train, xml_val, xml_test}')
