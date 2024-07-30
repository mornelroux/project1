import cv2
import numpy as np

class ImageEditor:
    def __init__(self):
        pass

    def fill_bounding_box(self,img, color, x, y, x_plus_w, y_plus_h,alpha=0.5):
        """
        Fills bounding boxes on the input image based on the provided arguments.

        Args:
            img (numpy.ndarray): The input image to draw the bounding box on.
            class_id (int): Class ID of the detected object.
            confidence (float): Confidence score of the detected object.
            x (int): X-coordinate of the top-left corner of the bounding box.
            y (int): Y-coordinate of the top-left corner of the bounding box.
            x_plus_w (int): X-coordinate of the bottom-right corner of the bounding box.
            y_plus_h (int): Y-coordinate of the bottom-right corner of the bounding box.
            alpha (float): The alpha value of the overlay.
        """
        #Convert from tensor format to int
        x = int(x.item())
        y = int(y.item())
        x_plus_w = int(x_plus_w.item())
        y_plus_h = int(y_plus_h.item())

        #Extract boundary box
        box = img[y:y_plus_h, x:x_plus_w]
        #Create a colored box with same dimensions
        filled_box = np.ones(box.shape, dtype=np.uint8) * np.array(color, dtype=box.dtype)
        #Overlay the colored box on the original
        img[y:y_plus_h, x:x_plus_w] = cv2.addWeighted(box, alpha, filled_box, 1 - alpha, 1) 

    def draw_bounding_box(self,img, color, x, y, x_plus_w, y_plus_h):
        """
        Draws bounding boxes on the input image based on the provided arguments.

        Args:
            img (numpy.ndarray): The input image to draw the bounding box on.
            class_id (int): Class ID of the detected object.
            confidence (float): Confidence score of the detected object.
            x (int): X-coordinate of the top-left corner of the bounding box.
            y (int): Y-coordinate of the top-left corner of the bounding box.
            x_plus_w (int): X-coordinate of the bottom-right corner of the bounding box.
            y_plus_h (int): Y-coordinate of the bottom-right corner of the bounding box.
        """

        x = int(x.item())
        y = int(y.item())
        x_plus_w = int(x_plus_w.item())
        y_plus_h = int(y_plus_h.item())
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 3)

    def draw_label(self,img, label,color, x, y):
        """
        Draws labels on the input image based on the provided arguments.

        Args:
            img (numpy.ndarray): The input image to draw the label on.
            label (str): The label to draw on the image.
            x (int): X-coordinate of the label.
            y (int): Y-coordinate of the label.
        """
        x = int(x.item())
        y = int(y.item())
        cv2.putText(img, label, (x+5, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,[255,255,255], 1)

    def add_title_to_image(self,img, title, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=1, box_color=(255, 255, 255), text_color=(0, 0, 0), padding=10,alpha=0.3):
        # Calculate text size
        (text_width, text_height), _ = cv2.getTextSize(title, font, font_scale, thickness)

        # Calculate box dimensions based on text size and padding
        box_width = text_width + 2 * padding
        box_height = text_height + 2 * padding

        # Calculate text position to center it on top of the image
        img_height, img_width = img.shape[:2]

        # Transperent box positioning
        x = ((img_width - box_width) // 2)
        y = 0
        x_plus_w = x + box_width
        y_plus_h = box_height

        # Extract boundary box
        box = img[y:y_plus_h, x:x_plus_w]
        # Create a colored box with same dimensions
        filled_box = np.ones(box.shape, dtype=np.uint8) * np.array(box_color, dtype=box.dtype)
        # Overlay the colored box on the original
        img[y:y_plus_h, x:x_plus_w] = cv2.addWeighted(box, 0.5, filled_box, 0.5, 1) 

        # Text Positioning (Origin at Right Bottom Corner)
        text_x = x + padding
        text_y = box_height - padding

        # Draw the title text on the image
        cv2.putText(img, title, (text_x, text_y), font, font_scale, text_color, thickness)

    def draw_line(self,img, x1, y1, x2, y2, id,line_clr=[0,0,0], corner_clr=[255,255,255],txt_clr=[0,0,0],corner_size=30):
        w = corner_size
        h = corner_size
        x1 = int(x1.item())
        y1 = int(y1.item())
        x2 = int(x2.item())
        y2 = int(y2.item())

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), line_clr, 1)

        # Top left corner
        cv2.line(img, (x1, y1), (x1 + w, y1), corner_clr, 2)
        cv2.line(img, (x1, y1), (x1, y1 + h), corner_clr, 2)

        # Top right corner
        cv2.line(img, (x2, y1), (x2 - w, y1), corner_clr, 2)
        cv2.line(img, (x2, y1), (x2, y1 + h), corner_clr, 2)

        # Bottom right corner
        cv2.line(img, (x2, y2), (x2 - w, y2), corner_clr, 2)
        cv2.line(img, (x2, y2), (x2, y2 - h), corner_clr, 2)

        # Bottom left corner
        cv2.line(img, (x1, y2), (x1 + w, y2), corner_clr, 2)
        cv2.line(img, (x1, y2), (x1, y2 - h), corner_clr, 2)

        text = f'ID:{str(id)}'
        cv2.putText(img, text,
                    (x1, y1 - 2),
                    0, 1 / 2, txt_clr,
                    thickness=1, lineType=cv2.FILLED)
