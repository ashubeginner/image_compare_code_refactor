# import the necessary packages
import codecs
import cv2
import imutils
import itertools
import json
import logging
import numpy as np
import os
import pickle
import pytesseract
import skimage
from datetime import date
from datetime import datetime
from datetime import timedelta
from jira import JIRA, JIRAError
from skimage.measure import compare_ssim
from skimage.feature import match_template
from sys import platform
debug = False


def compare(folder, datum_image="imageB.png", latest_image="imageA.png",
            masked_datum="imageB_masked.png"):

    datum_image, gray_datum_image, latest_image, gray_latest_image = preprocess_images(folder=folder,
                                                                                       datum_image=datum_image,
                                                                                       latest_image=latest_image)
    contours = extract_contours(gray_datum_image, gray_latest_image)

    rectangles = extract_regions_of_interest(contours=contours, gray_datum_image=gray_datum_image,
                                             gray_latest_image=gray_latest_image)

    jira_ticket = None

    if len(rectangles) > 0:
        merged_rectangles = combine_boxes(boxes=rectangles)
        new_text = extract_text(boxes=rectangles, image=gray_latest_image)
        diff_text = save_text(folder=folder, text=new_text)

        before_and_after_png_path = save_images(folder=folder, datum_image=datum_image, latest_image=latest_image)

        jira_ticket = {'summary': "Website change detected at {}".format(folder.split(os.sep)[-1]),
                       'description': "Changes detected at: {}\nPlease see the attachments for details.".format(folder),
                       'attachments': [diff_text, before_and_after_png_path, "{}{}{}".format(folder, os.sep, masked_datum)]}

    logging.info("\tChange detected: {}".format(len(rectangles) > 0))

    return jira_ticket


def save_images(folder, datum_image, latest_image):

    # save the output image
    diff_png = "{}{}{}".format(folder, os.sep, "diff.png")
    logging.debug("Diff image saved to {}".format(diff_png))

    cv2.imwrite(diff_png, latest_image)
    before_and_after_png = "{}{}{}".format(folder, os.sep, "before_and_after.png")

    logging.info("\tBefore and after png saved to: {}".format(before_and_after_png))
    cv2.imwrite(before_and_after_png, tile_images(left_image=datum_image, right_image=latest_image))

    return before_and_after_png


def save_text(folder, text, filename="diff.txt"):

    path = "{}\{}".format(folder, filename)
    f = codecs.open(path, 'w', encoding='utf8')

    for text in text:
        logging.debug("Extracted the following text from: [{}]".format(text))
        f.write(u'\n=======================================================================\n')
        f.write(text)

    f.flush()
    f.close()

    return path


def extract_regions_of_interest(contours, gray_datum_image, gray_latest_image, threshold=0.75,
                                contour_area_threshold=100):
    count = 0
    rectangles = []

    logging.debug("Processing {} contours to process".format(len(contours)))
    for contour in contours:
        count = count + 1
        logging.debug("Processing contour {} of {}".format(count, len(contours)))

        # compute the bounding box of the contour for the region of interest
        (x, y, w, h) = cv2.boundingRect(contour)
        region_of_interest = gray_latest_image[y:y + h, x:x + w]

        # extract text from region of interest
        text_of_interest = pytesseract.image_to_string(region_of_interest)

        if len(text_of_interest) <= 0:
            logging.debug("No text, so let's skip this contour.....")
            continue

        logging.debug("Found text: {}".format(text_of_interest))

        # search for the region of interest in the original image
        loc = skimage.feature.match_template(gray_datum_image, region_of_interest, pad_input=False,
                                             mode='constant', constant_values=0)

        match = np.where(loc >= threshold)
        logging.debug("correlation factor: {}".format(np.sum(match) / (loc.shape[0] * loc.shape[1])))

        if np.sum(match) / (loc.shape[0] * loc.shape[1]) <= 0 and h * w > contour_area_threshold:
            rectangles.append([x, y, x + w, y + h])

    return rectangles


def preprocess_images(folder, datum_image, latest_image):

    datum_image, latest_image = load_images(folder=folder, datum_image=datum_image, latest_image=latest_image)
    datum_image, latest_image = make_image_size_identical(datum_image=datum_image, latest_image=latest_image)

    # Convert the images to gray scale
    gray_datum_image = cv2.cvtColor(datum_image, cv2.COLOR_BGR2GRAY)
    gray_latest_image = cv2.cvtColor(latest_image, cv2.COLOR_BGR2GRAY)

    return datum_image, gray_datum_image, latest_image, gray_latest_image


def load_images(folder, datum_image, latest_image):

    # load the two input images
    logging.info("\tLoading datum image: {}{}{}".format(folder, os.sep, datum_image))
    datum_image = cv2.imread("{}{}{}".format(folder, os.sep, datum_image))

    logging.info("\tLoading latest scrape: {}{}{}".format(folder, os.sep, latest_image))
    latest_image = cv2.imread("{}{}{}".format(folder, os.sep, latest_image))

    return datum_image, latest_image


def extract_contours(gray_datum_image, gray_latest_image):
    # Check the structural similarity score
    (score, diff) = compare_ssim(gray_datum_image, gray_latest_image, full=True)
    diff = (diff * 255).astype("uint8")
    logging.debug("SSIM: {}".format(score))

    # Threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logging.debug(len(contours))
    contours = contours[0] if imutils.is_cv2() else contours[1]

    for contour in contours:
        logging.debug("extract_contours: {}".format(contour))

    return contours


def make_image_size_identical(datum_image, latest_image, max_height=5000):

    (H_datum, W_datum, C_datum) = datum_image.shape
    (H_latest, W_latest, C_latest) = latest_image.shape

    logging.debug("datum_image.shape: [{}, {}]".format(W_datum, H_datum))
    logging.debug("latest_image.shape: [{}, {}]".format(W_latest, H_latest))

    # Note: cv2.resize didn't work because it stretches images
    if H_datum > max_height and H_latest > max_height:
        datum_image = datum_image[:max_height, :W_datum]
        latest_image = latest_image[:max_height, :W_datum]
    elif H_datum * W_datum < H_latest * W_latest:
        latest_image = latest_image[:H_datum, :W_datum, :C_datum]
    else:
        datum_image = datum_image[:H_latest, :W_latest, :C_latest]

    logging.debug("datum_image.shape: [{}, {}]".format(W_datum, H_datum))
    logging.debug("latest_image.shape: [{}, {}]".format(W_latest, H_latest))

    if datum_image.shape != latest_image.shape:
        logging.error("Shape mismatch d=[{}] l=[{}]".format(datum_image.shape, latest_image.shape))

    return datum_image, latest_image


def extract_text(boxes, image):
    text = []

    for box in boxes:
        pixels = (box[3] - box[1]) * (box[2] - box[0])
        factor = (2 ** 31) / pixels

        # todo: high factors cause tesseract to crash
        factor = 4

        if factor > 20:
            factor = 20

        logging.debug("extract_text: Inflating image by a factor of {}".format(factor))
        logging.debug("width: {}".format(int((box[2] - box[0]))))
        logging.debug("height: {}".format(int((box[3] - box[1]))))
        logging.debug("total pixels: {}".format(int((box[2] - box[0]) * factor) * int((box[3] - box[1]) * factor)))
        logging.debug("max pixels: {}".format(2 ** 31))
        region_of_interest = image[box[1]:box[3], box[0]:box[2]]
        resized_region_of_interest = cv2.resize(region_of_interest,
                                                (int((box[2] - box[0]) * factor), int((box[3] - box[1]) * factor)),
                                                interpolation=cv2.INTER_CUBIC)
        text.append("{}".format(pytesseract.image_to_string(resized_region_of_interest)))

    return text


def tile_images(left_image, right_image):
    return np.concatenate((left_image, right_image), axis=1)


def union(a, b):
    x1 = min(a[0], b[0])
    y1 = min(a[1], b[1])
    x2 = max(a[2], b[2])
    y2 = max(a[3], b[3])
    return x1, y1, x2, y2


def combine_boxes(boxes, blur=5):
    ignore = []
    previous_length = 0
    current_length = len(boxes)
    iterations = []
    recurse = True

    while current_length != previous_length and recurse:
        for boxa, boxb in itertools.combinations(boxes, 2):
            if intersecting_squares((boxa[0] - blur, boxa[1] - blur), (boxa[2] + blur, boxa[3] + blur),
                                    (boxb[0] - blur, boxb[1] - blur), (boxb[2] + blur, boxb[3] + blur)):
                logging.debug("boxa:{}".format(boxa))
                logging.debug("boxb:{}".format(boxb))

                box_ab = union(boxa, boxb)
                if boxa in boxes:
                    logging.debug("removing boxa: {}".format(boxa))
                    boxes.remove(boxa)
                    ignore.append(boxa)

                if boxb in boxes:
                    logging.debug("removing boxb: {}".format(boxb))
                    boxes.remove(boxb)
                    ignore.append(boxb)

                if box_ab not in boxes:
                    logging.debug(("boxes: {}".format(boxes)))
                    boxes.append(box_ab)
                    logging.debug(("appending boxab: {}".format(box_ab)))

        iterations.append(len(boxes))

        if len(iterations) > 2:
            if iterations[-1] == iterations[-2] == iterations[-3]:
                recurse = False

        logging.debug("iterations={}".format(iterations))

    return np.array(boxes).astype('int')


def between(lower, upper, datum):
    return upper >= datum >= lower


def intersecting(r1_bottom_left, r1_top_right, r2_bottom_left, r2_top_right):
    x_lower = r1_bottom_left[0]
    x_upper = r1_top_right[0]
    y_lower = r1_bottom_left[1]
    y_upper = r1_top_right[1]

    return ((between(lower=x_lower, upper=x_upper, datum=r2_bottom_left[0])
            or between(lower=x_lower, upper=x_upper, datum=r2_top_right[0]))
            and between(lower=y_lower, upper=y_upper, datum=r2_bottom_left[1])
            or between(lower=y_lower, upper=y_upper, datum=r2_top_right[1]))


def intersecting_squares(r1_bottom_left, r1_top_right, r2_bottom_left, r2_top_right):

    return (intersecting(r1_bottom_left, r1_top_right, r2_bottom_left, r2_top_right)
            or intersecting(r2_bottom_left, r2_top_right, r1_bottom_left, r1_top_right))


def create_jira_ticket(summary, attachments, description, url, issuetype='Task', username='xxxxxxxxxxxxxxxxxx',
                       api_key='xxxxxxxxxxxxxxxxxxxxxxxxxx', server='https://xxxxxxx.atlassian.net'):
    new_issue = {}
    jira_options = {'server': server}
    jira_client = JIRA(options=jira_options, basic_auth=(username, api_key))

    web_monitor_project = {'key': 'WM', 'name': 'Web Monitor', 'id': '12615'}

    issue_dict = {
        'project': {'id': web_monitor_project['id']},
        'summary': summary,
        'description': description,
        'issuetype': {'name': issuetype},
        'customfield_11833': url
    }

    try:
        new_issue = jira_client.create_issue(fields=issue_dict)
    except JIRAError as e:
        logging.error("Could not create Jira ticket [{}]".format(issue_dict))
        logging.error("{}".format(e))

    for attachment in attachments:
        if os.path.exists(attachment):
            try:
                jira_client.add_attachment(issue=new_issue, attachment=attachment)
            except JIRAError as e:
                logging.error("Could not add attachment [{}]".format(attachment))
                logging.error("{}".format(e))
        else:
            logging.error("Could find file to attach to Jira Ticket: [{}]".format(attachment))

    logging.info("\tCreated jira ticket: [{}]".format(new_issue.key))

    return


def append_contours_to_masks(contours, mask_path):

    with open(str(mask_path, 'rb')) as pkl:
        diff_contours = pickle.load(pkl)

    diff_contours.append(contours)

    with open("{}/diff_contours.pkl".format(mask_path), 'wb') as pkl:
        pickle.dump(diff_contours, pkl)

    return 1


def process_alerts(root,
                   json_alerts="website_changes_log.json", back_date=date.today(), skip=0):

    logging.info("Processing root folder: {}".format(root))

    count = 1
    jira_count = 0

    website_changes_log = open("{}{}{}".format(root, os.sep, json_alerts), "r")

    for change in website_changes_log:

        change = json.loads(change)
        url = change["url"]
        folder = convert_url_to_folder(url)

        scan_date = change["Date"]
        scan_date = datetime.strptime(scan_date, '%Y%m%d-%H%M')

        if scan_date.date() < back_date:
            continue

        logging.info("[{}] Processing folder: {}".format(count, "{}{}{}".format(root, os.sep, folder)))

        datum_image = None
        latest_image = None

        if os.path.isdir("{}{}{}".format(root, os.sep, folder)) and count >= skip:
            for file in os.listdir("{}{}{}".format(root, os.sep, folder)):
                if file == 'imageB.png':
                    # imageB is the datum
                    datum_image = file
                elif file == 'imageA.png':
                    # imageA is the latest image
                    latest_image = file

            # todo: remove
            if debug:
                datum_image = "imageB.png"
                latest_image = "imageB.png"

            if datum_image is not None and latest_image is not None:
                jira_ticket_details = compare(folder="{}{}{}".format(root, os.sep, folder))

                if jira_ticket_details:
                    jira_count = jira_count + 1

        count = count + 1

    logging.info("Processed a total of [{}] changes and WOULD HAVE created [{}] jira tickets".format(count, jira_count))


def convert_url_to_folder(url):
    folder = url.replace(".", "")
    folder = folder.replace("https:", "")
    folder = folder.replace("http:", "")
    folder = folder.replace("/", "")
    folder = folder.replace("?", "")

    return folder


if __name__ == '__main__':

    file_root = "S:\\A path\\to\\a\\file\\system"

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    if debug:
        process_alerts(root=file_root,
                       back_date=date.today() - timedelta(days=6))
    else:
        process_alerts(root=file_root,
                       back_date=date.today() - timedelta(days=9))