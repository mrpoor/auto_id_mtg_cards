import os

import cv2
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from core.cross_correlate import cross_correlate_all_set_images


def get_confusion_matrix():
    thresh_image_path = 'thresh_resize_images'
    list_set_images = sorted(os.listdir(thresh_image_path))
    all_rows = []
    index_dict = dict()
    for index, img in enumerate(list_set_images):
        index_dict[index] = img
        set_image = cv2.imread(os.path.join(thresh_image_path, img), cv2.IMREAD_GRAYSCALE )
        result_dict = cross_correlate_all_set_images(set_image)
        all_rows.append(result_dict)
    confusion_matrix_df = pd.DataFrame(all_rows)
    confusion_matrix_df.rename(index=index_dict, inplace=True)
    return confusion_matrix_df
