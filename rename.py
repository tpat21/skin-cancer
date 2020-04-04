import os


def rename_files(DIR):
    path = os.chdir(DIR)
    i = 0
    for file in os.listdir(DIR):
        if("benign" in DIR):
            new_file_name = "benign{}.jpg".format(i)
            os.rename(file,new_file_name)
            i = i+1
        else:
            new_file_name = "malignant{}.jpg".format(i)
            os.rename(file,new_file_name)
            i = i+1


new_benign_test = rename_files("/Users/tpat/PycharmProjects/skin-cancer/data/test/benign")
new_malign_test = rename_files("/Users/tpat/PycharmProjects/skin-cancer/data/test/malignant")

new_malign_train = rename_files("/Users/tpat/PycharmProjects/skin-cancer/data/train/malignant")
new_benign_train = rename_files("/Users/tpat/PycharmProjects/skin-cancer/data/train/benign")


