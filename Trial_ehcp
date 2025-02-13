import boto3
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_bytes
import easyocr
from PIL import Image
import io
import re
import csv
from collections import defaultdict
from datetime import datetime
import PyPDF2
import spacy
import traceback
import json
from datetime import timedelta
import textract

# Load spaCy model
nlp = spacy.load("en_core_web_lg")

def append_to_s3_csv(bucket, key, data):
    """Append data to CSV file in S3"""
    s3_client = boto3.client('s3')
    
    try:
        # Try to get existing file
        try:
            response = s3_client.get_object(Bucket=bucket, Key=key)
            existing_content = response['Body'].read().decode('utf-8')
            existing_file = io.StringIO(existing_content)
            reader = csv.DictReader(existing_file)
            fieldnames = reader.fieldnames
        except s3_client.exceptions.NoSuchKey:
            # File doesn't exist, use data's keys as fieldnames
            fieldnames = data.keys()
        
        # Create new CSV in memory
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        
        # Write header only for new files
        if not existing_content:
            writer.writeheader()
        
        # Write the new data
        writer.writerow(data)
        
        # Upload to S3
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=output.getvalue().encode('utf-8'),
            ContentType='text/csv'
        )
    except Exception as e:
        print(f"Error appending to CSV in S3: {str(e)}")
        raise

def preprocess_image_s3(image_bytes):
    """Preprocess image from S3 bytes"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    coords = np.column_stack(np.where(binary > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated

def extract_text_from_pdf_s3(s3_object):
    """Extract text from PDF stored in S3 using both direct extraction and OCR"""
    try:
        # First try direct text extraction
        pdf_bytes = io.BytesIO(s3_object.read())
        reader = PyPDF2.PdfReader(pdf_bytes)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        
        # Check if we got sufficient text
        word_count = len(text.strip().split())
        if word_count >= 100:
            return text.strip()
        
        print(f"Direct extraction yielded insufficient text ({word_count} words). Switching to OCR.")
        
        # Reset file pointer for OCR processing
        s3_object.seek(0)
        
        # Convert PDF to images
        images = convert_from_bytes(s3_object.read())
        
        # Initialize OCR readers
        easyocr_reader = easyocr.Reader(['en'])
        
        full_text = ""
        
        for image in images:
            # Convert PIL Image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Preprocess image
            preprocessed = preprocess_image_s3(img_byte_arr)
            
            # Try Tesseract OCR
            try:
                tesseract_text = pytesseract.image_to_string(preprocessed)
            except Exception as e:
                print(f"Tesseract OCR failed: {str(e)}. Falling back to EasyOCR.")
                tesseract_text = ""
            
            # Apply EasyOCR
            easyocr_result = easyocr_reader.readtext(preprocessed)
            easyocr_text = ' '.join([res[1] for res in easyocr_result])
            
            # Use the longer text
            page_text = tesseract_text if len(tesseract_text) > len(easyocr_text) else easyocr_text
            full_text += page_text + "\n\n"
        
        return full_text.strip()
        
    except Exception as e:
        print(f"Error in PDF extraction: {str(e)}")
        traceback.print_exc()
        return ""

def extract_name(text):
    """Extract student name from text using NLP"""
    # Focus on the first 1000 characters
    text = text[:1000]
    
    # Use spaCy's named entity recognition
    doc = nlp(text)
    person_entities = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    
    if person_entities:
        return clean_name(person_entities[0])
    
    # Fallback patterns if NLP fails
    patterns = [
        r"([\w\s-]+)\s*\n\s*Education,\s+Health\s+and\s+Care\s+Plan",
        r"Surname:\s*(\w+)\s*Forenames:\s*(\w+)",
        r"([\w\s-]+)\s*\n\s*Date of Birth:",
        r"([\w\s-]+)(?:\s*\([^)]+\))?\s*'s\s*Education\s+Health\s+&\s+Care\s+Plan",
        r"Child\s+Name:\s*([\w\s-]+)",
        r"([\w\s-]+)'s\s+Plan",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            if len(match.groups()) == 2 and 'Surname' in pattern:
                name = f"{match.group(2)} {match.group(1)}"
            else:
                name = match.group(1)
            return clean_name(name)
    
    return "Unknown"

def extract_upn(text):
    """Extract UPN from text"""
    patterns = [
        r'\b[A-Za-z]\d{12}\b',
        r'UPN:?\s*([A-Za-z]\d{12})',
        r'Unique\s+Pupil\s+Number:?\s*([A-Za-z]\d{12})',
        r'Student\s+ID:?\s*([A-Za-z]\d{12})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            upn = match.group(1) if len(match.groups()) > 0 else match.group(0)
            if upn and len(upn) == 13:
                return upn
    
    return "Unknown"

def extract_categories_and_subcategories(text):
    """Extract and categorize special needs from text"""
    # Keep existing categories and subcategories logic
    subcategories = {
            "Autism": [
                "autism", "autistic", "asd", "asc", "aspergers"
            ],
            
            "ADHD": [
                "adhd", "attention deficit hyperactivity disorder"
            ],
            
            "DCD/Dyspraxia": [
                "dyspraxia", "dyspraxic", "dcd", "developmental coordination disorder"
            ],
            
            "Dyscalculia": [
                "dyscalculia", "dyscalculic"
            ],
            
            "Dyslexia": [
                "dyslexia", "dyslexic"
            ],
            
            "OCD": [
                "ocd", "obsessive compulsive disorder"
            ],
            
            "DLD": [
                "dld", "developmental language disorder", "dysphasia", "dysphasic"
            ],
            
            "Anxiety": [
                "anxiety", "anxiety disorder"
            ],
            
            "Depression": [
                "depression", "depressive", "clinical depression"
            ],
            
            "Trauma/PTSD": [
                "ptsd", "post traumatic stress disorder", "trauma"
            ],
            
            "ODD": [
                "odd", "oppositional defiant disorder"
            ],
            
            "Learning_Difficulties": [
                "learning disability", "subnormal", "dyscognitive"
            ],
            
            "SLCN": [
                "slcn", "speech language and communication needs", "speech and language", 
                "speech delay", "language delay", "communication difficulty",
                "speech disorder", "language disorder", "communication disorder"
            ]
        }
    
    word_count = defaultdict(int)
    
    for category, keywords in subcategories.items():
        category_count = sum(1 for keyword in keywords if keyword in text.lower())
        if category_count > 0:
            word_count[category] = category_count
    
    categories = {category: count for category, count in word_count.items() if count > 0}
    
    return categories, word_count

def get_next_id_dynamo():
    """Generate next EHCP ID using DynamoDB atomic counter"""
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('ehcp_counter')
    response = table.update_item(
        Key={'counter_name': 'ehcp'},
        UpdateExpression='ADD counter_value :inc',
        ExpressionAttributeValues={':inc': 1},
        ReturnValues='UPDATED_NEW'
    )
    return f"E{response['Attributes']['counter_value']:05d}"

def store_in_dynamodb(ehcp_info, categories):
    """Store EHCP data in DynamoDB tables"""
    dynamodb = boto3.resource('dynamodb')
    ehcp_table = dynamodb.Table('ehcp_records')
    categories_table = dynamodb.Table('ehcp_categories')
    
    ehcp_table.put_item(Item=ehcp_info)
    for category in categories:
        categories_table.put_item(Item=category)

def process_ehcp_file_s3(s3_object, school, local_authority, ehcp_id):
    """Process EHCP file from S3"""
    try:
        text = extract_text_from_pdf_s3(s3_object)
        if not text:
            return None, []

        name = extract_name(text)
        upn = extract_upn(text)
        categories, keyword_counts = extract_categories_and_subcategories(text)
        
        ehcp_info = {
            'EHCP_ID': ehcp_id,
            'Name': name,
            'UPN': upn,
            'School': school,
            'LocalAuthority': local_authority,
            'EHCP_Date': datetime.now().strftime('%Y-%m-%d'),
            'ProcessedDate': datetime.now().strftime('%Y-%m-%d')
        }
        
        categories_list = [
            {
                'EHCP_ID': ehcp_id,
                'Category': category,
                'KeywordCount': count
            }
            for category, count in keyword_counts.items() if count > 0
        ]
        
        return ehcp_info, categories_list
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        traceback.print_exc()
        return None, []

def lambda_handler(event, context):
    """AWS Lambda handler"""
    try:
        bucket = event['Records'][0]['s3']['bucket']['name']
        file_key = event['Records'][0]['s3']['object']['key']
        
        s3_client = boto3.client('s3')
        s3_object = s3_client.get_object(Bucket=bucket, Key=file_key)['Body']
        
        path_parts = file_key.split('/')
        school = path_parts[0]
        local_authority = path_parts[1]
        
        ehcp_id = get_next_id_dynamo()
        ehcp_info, categories = process_ehcp_file_s3(s3_object, school, local_authority, ehcp_id)
        
        if ehcp_info:
            store_in_dynamodb(ehcp_info, categories)
            results_bucket = 'your-results-bucket-name'
            append_to_s3_csv(results_bucket, 'ehcp_info.csv', ehcp_info)
            for category in categories:
                append_to_s3_csv(results_bucket, 'categories.csv', category)
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': f"Successfully processed EHCP {ehcp_id}",
                    'ehcp_id': ehcp_id
                })
            }
        
        return {
            'statusCode': 400,
            'body': json.dumps({
                'message': "Failed to process EHCP",
                'file_key': file_key
            })
        }
            
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        traceback.print_exc()
        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': f"Error: {str(e)}",
                'file_key': file_key
            })
        }