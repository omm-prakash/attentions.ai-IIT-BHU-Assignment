import torch
from PIL import Image
import os
import requests
import arxiv
from pdf2image import convert_from_path
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import pandas as pd
from tqdm import tqdm
import warnings
import tempfile
import platform
import subprocess
warnings.filterwarnings('ignore')

class ArxivPDFProcessor:
    def __init__(self, output_directory, model_name="Salesforce/blip2-opt-2.7b"):
        self.output_directory = output_directory
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)
        os.makedirs(os.path.join(output_directory, "extracted_images"), exist_ok=True)
        
        print(f"Initializing BLIP-2 model on {self.device}...")
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.model.to(self.device)
        self.results = []

    def check_poppler(self):
        system = platform.system().lower()
        
        if system == 'windows':
            # Check if poppler is in PATH
            from shutil import which
            if which('pdftoppm') is None:
                print("Poppler not found in PATH.")
                print("Please download poppler from: http://blog.alivate.com.au/poppler-windows/")
                print("Extract it and add the bin/ directory to your PATH")
                return False
        
        elif system == 'linux':
            try:
                subprocess.run(['pdftoppm', '-v'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return True
            except FileNotFoundError:
                print("Poppler not found. Installing...")
                try:
                    subprocess.run(['sudo', 'apt-get', 'update'])
                    subprocess.run(['sudo', 'apt-get', 'install', '-y', 'poppler-utils'])
                    return True
                except:
                    print("Failed to install poppler. Please install manually: sudo apt-get install poppler-utils")
                    return False
        
        elif system == 'darwin':  # macOS
            try:
                subprocess.run(['pdftoppm', '-v'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return True
            except FileNotFoundError:
                print("Poppler not found. Installing...")
                try:
                    subprocess.run(['brew', 'install', 'poppler'])
                    return True
                except:
                    print("Failed to install poppler. Please install manually: brew install poppler")
                    return False
        
        return True

    def download_arxiv_paper(self, arxiv_url):
        print(f"Downloading paper from {arxiv_url}")
        
        # Extract arxiv ID from URL
        arxiv_id = arxiv_url.split('/')[-1]
        
        try:
            # Search for the paper
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(search.results())
            
            # Download the paper
            temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            paper.download_pdf(filename=temp_pdf.name)
            
            print(f"Successfully downloaded: {paper.title}")
            return temp_pdf.name, paper.title
            
        except Exception as e:
            print(f"Error downloading paper: {str(e)}")
            return None, None

    def extract_images_from_pdf(self, pdf_path):
        try:
            print("Converting PDF to images...")
            images = convert_from_path(
                pdf_path,
                dpi=200,
                fmt='jpg',
                thread_count=4,
                poppler_path=None  # Will use system's poppler installation
            )
            return [(img, idx+1) for idx, img in enumerate(images)]
        except Exception as e:
            print(f"Error converting PDF to images: {str(e)}")
            return []

    def generate_caption(self, image):
        try:
            inputs = self.processor(image, return_tensors="pt").to(self.device, torch.float16)
            generated_ids = self.model.generate(**inputs, max_length=50)
            caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            print(f"Error generating caption: {str(e)}")
            return "Caption generation failed"

    def process_arxiv_paper(self, arxiv_url):
        # First check if poppler is properly installed
        if not self.check_poppler():
            print("Please install poppler first and try again.")
            return

        # Download PDF
        pdf_path, paper_title = self.download_arxiv_paper(arxiv_url)
        if not pdf_path:
            print("Failed to download PDF")
            return
        
        try:
            # Extract images from PDF
            images = self.extract_images_from_pdf(pdf_path)
            print(f"Extracted {len(images)} pages")
            
            # Process each image
            for image, page_num in tqdm(images, desc="Processing pages"):
                try:
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    caption = self.generate_caption(image)
                    
                    arxiv_id = arxiv_url.split('/')[-1]
                    image_filename = f"{arxiv_id}_page{page_num}.jpg"
                    image_path = os.path.join(
                        self.output_directory, 
                        "extracted_images", 
                        image_filename
                    )
                    image.save(image_path)
                    
                    self.results.append({
                        'arxiv_id': arxiv_id,
                        'paper_title': paper_title,
                        'page_number': page_num,
                        'image_path': image_path,
                        'caption': caption
                    })
                    
                except Exception as e:
                    print(f"Error processing page {page_num}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
        
        finally:
            # Clean up temporary PDF file
            try:
                os.unlink(pdf_path)
            except:
                pass
            
        self.save_results()

    def save_results(self):
        if self.results:
            df = pd.DataFrame(self.results)
            csv_path = os.path.join(self.output_directory, 'image_analysis_results.csv')
            df.to_csv(csv_path, index=False)
            print(f"\nResults saved to {csv_path}")
        else:
            print("\nNo results to save")



def main():
    # arXiv URL and output directory
    arxiv_url = "https://arxiv.org/pdf/2004.07606"
    output_dir = "arxiv_output"
    
    # Initialize processor and process paper
    processor = ArxivPDFProcessor(output_dir)
    processor.process_arxiv_paper(arxiv_url)

if __name__ == "__main__":
    main()