import sys, os
import cv2
import com_files as comf

if __name__ == '__main__':

    save_path      = "/home/leo/myhome/WIDER_train/"
    root_path      = "/home/leo/myhome/WIDER_train/images"
    label_out_path = "/home/leo/myhome/WIDER_train/labels_face"

    label_path = os.path.join(root_path, 'label.txt')
    if not os.path.isfile(label_path):
        print('Missing label.txt file.')
        exit(1)

    afile = open(label_path)
    label_lines = afile.readlines()
    afile.close()

    #/home/leo/myhome/WIDER_train/images/0--Parade/xxxxx.jpg
    #/home/leo/myhome/WIDER_train/labels_face0--Parade/xxxxx.txt

    #label_lines format:
    # 0--Parade/0_Parade_marchingband_1_849.jpg
    # 1
    # 449 330 122 149 0 0 0 0 0 0 

    image_files = []
    for i , v in enumerate(label_lines):
        if ".jpg" in v:
            filename = v.strip()
            # print(" a->", label_lines[i+1].strip())
            num = int(label_lines[i+1].strip())
            # print(" b->", i, filename, num)

            faces = []
            for seq in range(1, num+1):
                item = label_lines[i+1+seq].strip()
                #print(" c->", seq, item)
                faces.append(item)

            img_file = os.path.join(root_path, filename)
            image_files.append(img_file)

            label_filename = os.path.join(label_out_path, filename)
            label_filename = label_filename.replace(".jpg", ".txt")
            comf.ensure_file_dir(label_filename)

            img = cv2.imread(img_file)
            height, width, _ = img.shape
            img = None
            afile = open(label_filename, "w+")
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
            # print(f"write labels:{label_filename}")
            # if i > 10:
            #     break

    outfilename = os.path.join(save_path, "wider_filelists.txt")
    outfile = open(outfilename, "w+")
    for each in image_files:
        filepath = os.path.abspath(each)
        outfile.write(filepath + "\n")
    outfile.close()

    print(f"output:{outfilename}")
    print(f"   num:{len(image_files)}")

