from langchain.schema import Document
import pandas as pd
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from src.entity.config_entity import DirPath
from src.exception import GeneralPractitionerException
from src.logger import logging
import os
import sys


class DataConverter:
     
    def __init__(self, file_path):
        try:
            self.file_path = file_path
            self.content = []
            self.page_num = []
            self.docs = []
            self.images = None
        except Exception as e:
            raise GeneralPractitionerException(e, sys)

    def convert_pdf_to_image(self, dpi=300):
        """
        Convert the PDF to images.
        
        Args:
        - dpi (int): Dots per inch for image conversion. Default is 300.
        """
        try:
            
            logging.info("Start converting pdf into images.")
            path = DirPath()
            output_path = path.get_images_dir()
        
            # Convert PDF to images
            self.images = convert_from_path(self.file_path, dpi=dpi)

            # Save each page as an image
            for i, page in enumerate(self.images):
                image_path = os.path.join(output_path, f'page_{i + 1}.png')
                page.save(image_path, 'PNG')
            logging.info("End converting pdf into images.")

        except Exception as e:
            raise GeneralPractitionerException(e, sys)        
    
    def convert_image_to_text(self):

        try:

            logging.info("Start converting images into text.")
            for i, image in enumerate(self.images):
                text = pytesseract.image_to_string(image)
                self.page_num.append(i + 1)
                self.content.append(text)
            logging.info("End converting images into text.")
            return self.page_num, self.content
        except Exception as e:
            raise GeneralPractitionerException(e, sys)


    def convert_text_to_doc(self):

        try:
            logging.info("Start converting text into Document Format.")
            data_list = []
            
            # Prepare data for Document creation
            data = pd.DataFrame({
                "page": self.page_num,
                "content": self.content
            })
            
            for index, row in data.iterrows():
                data_list.append({
                    'page': row['page'],
                    'content': row['content']
                })
            
            # Create Document objects
            for entry in data_list:
                metadata = {
                    "source": self.file_path, 
                    "file_path": self.file_path, 
                    'total_pages': len(self.page_num), 
                    'format': 'PDF 1.4', 
                    'title': '', 
                    "page": entry['page']
                }
                doc = Document(page_content=entry['content'], metadata=metadata)
                self.docs.append(doc)
            logging.info("End of converting text into Document Format.")
            return self.docs    
            # print(self.docs)
        except Exception as e:
            raise GeneralPractitionerException(e, sys)

if __name__ == '__main__':

    pdf_processor = DataConverter(file_path="uploads\\7210843.pdf")

    pdf_processor.convert_pdf_to_image()

    pdf_processor.convert_image_to_text()

    pdf_processor.convert_text_to_doc()

