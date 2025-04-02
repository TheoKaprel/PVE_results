#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt

def main():
    print(args)

    fig,ax = plt.subplots()
    trial = 0
    with open(args.log) as file:
        for line in file.readlines():
            if line[:10]=="Epoch 1/20":
                errors = []
            if line[:19]=="Current test error:":
                splt = line.split(' ')
                errors.append(float(splt[-1]))
            if "finished" in line:
                ax.plot(errors, label = f"{trial}")
                trial+=1
    plt.legend()
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log")
    args = parser.parse_args()

    main()
