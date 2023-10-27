#!/usr/bin/env python3

import argparse
import itk
import json
import numpy as np

def main():
    print(args)

    labels=itk.imread(args.labels)
    labels_np = itk.array_from_image(labels)

    labels_json = open(args.json).read()
    labels_json = json.loads(labels_json)

    src=np.zeros((labels_np.shape))

    src[labels_np==labels_json["background"]]=args.bg_act

    lspheres=["iec_sphere_10mm",
        "iec_sphere_13mm",
        "iec_sphere_17mm",
        "iec_sphere_22mm",
        "iec_sphere_28mm",
        "iec_sphere_37mm"]

    for sp in lspheres:
        src[labels_np==labels_json[sp]]=args.src_act

    src_itk=itk.image_from_array(src)
    src_itk.SetSpacing(labels.GetSpacing())
    src_itk.SetOrigin(labels.GetOrigin())
    src_itk.SetDirection(labels.GetDirection())
    itk.imwrite(src_itk,args.output)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels")
    parser.add_argument("--json")
    parser.add_argument("--src_act", type=float)
    parser.add_argument("--bg_act", type=float)
    parser.add_argument("-o","--output")
    args = parser.parse_args()

    main()
