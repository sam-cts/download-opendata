import argparse
import pandas as pd
import os, sys
from tqdm import tqdm

def get_label_meta(labels, fpath):
    print(labels)
    meta_labels = []
    myfile = open(fpath, 'r')
    while myfile:
        line  = myfile.readline().replace('\n', '')      
        if line == "":
            break
        data = line.split(",")
        if data[1] in labels:
            meta_labels.append(data[0])
    myfile.close()
    return meta_labels

def get_image_label(labels, path):
    df = pd.read_csv(path)
    return df[df['LabelName'].isin(labels)]['ImageID'].tolist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--labels',
        required=True,
        nargs='+',
        type=str,
        default=None,
        help='list of label to download')
    parser.add_argument(
        '--dir',
        type=str,
        default="./annotation",
        help='list of label to download')
    parser.add_argument(
        '--output',
        type=str,
        default="images.txt",
        help='Directory where to store the image list.')

    args = parser.parse_args()
    if not os.path.exists(args.dir):
        print("input directory not found.")
        sys.exit()
    
    meta_labels = get_label_meta(args.labels, os.path.join(args.dir, "class-descriptions-boxable.csv"))
    print(meta_labels)

    fileNames = ['oidv6-train-annotations-bbox.csv', "validation-annotations-bbox.csv", "test-annotations-bbox.csv"]
    subsets = ["train", "validation", "test"]

    outFile = open(args.output, "w+")
    for i, fileName in enumerate(tqdm(fileNames)):
        imageIDs = get_image_label(meta_labels, os.path.join(args.dir, fileName))
        for imgID in imageIDs:
            outFile.write(f"{subsets[i]}/{imgID}\n")
        
    