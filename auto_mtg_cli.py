import os
import operator

from core.background_subtract import find_card_using_subtraction
from core.confusion_matrix import get_confusion_matrix
from core.cross_correlate import cross_correlate_all_set_images, thresh_all_set_templates
from core.mtg_api import get_all_cards_in_set, make_set_image_each_set
from core.preprocess import PreprocessorImg


def run_set_compare(path_to_original_image):
    pre_process = PreprocessorImg(path_to_original_image)
    crop_set_image = pre_process.mtg_thres_set_image
    thresh_all_set_templates(crop_set_image)
    cross_correlate_dict = cross_correlate_all_set_images(crop_set_image)
    max_tuple = max(cross_correlate_dict.items(), key=operator.itemgetter(1))
    print(max_tuple)
    return max_tuple[0].split('.')[0]


def run_card_compare(path_to_original_image):
    set_code = run_set_compare(path_to_original_image)
    pre_process = PreprocessorImg(path_to_original_image)
    find_card_using_subtraction(pre_process.mtg_just_card, set_code)


def run_confusion_matrix(path_to_original_image):
    make_set_image_each_set()
    pre_process = PreprocessorImg(path_to_original_image)
    crop_set_image = pre_process.mtg_thres_set_image
    thresh_all_set_templates(crop_set_image)
    confusion_matrix = get_confusion_matrix()
    print(confusion_matrix)


if __name__ == '__main__':
    run_confusion_matrix(os.path.join('mtg_test_photos', 'mtg_1.jpg'))
    # run_set_compare(os.path.join('mtg_test_photos', 'mtg_1.jpg'))
    # run_card_compare(os.path.join('mtg_test_photos', 'mtg_4.jpg'))