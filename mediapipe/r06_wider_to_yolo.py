import sys, os
import cv2

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Run command: python3 val2yolo.py /path/to/original/widerface [/path/to/save/widerface/val]')
        exit(1)

    root_path = sys.argv[1]
    save_path = sys.argv[2]

    label_path = os.path.join(root_path, 'label.txt')
    if not os.path.isfile(label_path):
        print('Missing label.txt file.')
        exit(1)

    afile = open(label_path)
    datas = afile.readlines()
    afile.close()

    #N:\myhome\WIDER_train\train\images\0--Parade\xxxxx.jpg
    #N:\myhome\WIDER_train\train\images\0--Parade\xxxxx.txt

    all_files = []
    for i , v in enumerate(datas):
        if ".jpg" in v:
            filename = v.strip()
            # print(" a->", datas[i+1].strip())
            num = int(datas[i+1].strip())
            # print(" b->", i, filename, num)

            faces = []
            for seq in range(1, num+1):
                item = datas[i+1+seq].strip()
                #print(" c->", seq, item)
                faces.append(item)

            filename = os.path.join(root_path,filename)

            all_files.append(filename)
            label_name = filename.replace(".jpg", ".txt")
            img = cv2.imread(filename)
            height, width, _ = img.shape
            img = None
            afile = open(label_name, "w+")
            # x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose
            #113 212 11 15 2 0 0 0 0 0
            for face in faces:
                items = face.split()
                x = float(items[0]) / float(width)
                y = float(items[1]) / float(height)
                w = float(items[2]) / float(width)
                h = float(items[3]) / float(height)
                blur = int(items[4])
                expression = int(items[5])
                illumination = int(items[6])
                invalid = int(items[7])
                occlusion = int(items[7])
                pose = int(items[7])
                cx  = x+w / 2.0
                cy  = y+h / 2.0
                #cxcywh
                yolo_fmt = f"1 {cx} {cy} {w} {h}\n"
                if invalid:
                    continue
                afile.write(yolo_fmt)
                #print(face, yolo_fmt),
            afile.close()
            #print(f"write:{label_name}")
            # if i > 10:
            #     break
    outfilename = os.path.join(save_path, "wider_face_filelists.txt")
    outfile = open(outfilename, "w+")
    for each in all_files:
        filepath = os.path.abspath(each)
        outfile.write(filepath + "\n")
    outfile.close()


    print(f"output:{outfilename}")
    print(f"   num:{len(all_files)}")
    
# python run01_to_yolo.py train/images/ .
