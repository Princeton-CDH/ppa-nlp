#!/usr/bin/env python

"""
setup:
 - in google cloud console, enable vision api for desired project
 - install gcloud cli (sdk) and login (this is now preferred to service accounts)
 - install python client: `pip install google-cloud-vision`
"""

import glob
import io
import os

from google.cloud import vision


IMAGE_DIR = "images"


def get_image_ocr():
    # Instantiate google vision client
    client = vision.ImageAnnotatorClient()

    for imagefile in glob.iglob("%s/*/*.jpg" % IMAGE_DIR):
        basepath = os.path.splitext(imagefile)[0]
        textfile = "%s.txt" % basepath
        jsonfile = "%s.json" % basepath

        # if text file already exists, skip
        if os.path.isfile(textfile):
            continue

        # if text file does not exist, request ocr

        # Load the image into memory
        with io.open(imagefile, "rb") as image_file:
            content = image_file.read()
        image = vision.Image(content=content)

        # Performs ocr and handwriting detection on the image file
        response = client.document_text_detection(image=image)

        # save plain text output to local file;
        # even if text is empty, create text file so we don't request again
        with open(textfile, "w") as textfilehandle:
            textfilehandle.write(response.full_text_annotation.text)

        # this generates a json string
        json_response = vision.AnnotateImageResponse.to_json(response)
        with open(jsonfile, "w") as jsonfilehandle:
            jsonfilehandle.write(json_response)

        if response.error.message:
            raise Exception(
                "{}\nFor more info on error messages, check: "
                "https://cloud.google.com/apis/design/errors".format(
                    response.error.message
                )
            )

        # stop processing after first image until we make decisions & refine
        break


if __name__ == "__main__":
    get_image_ocr()
