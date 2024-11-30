import email, smtplib, ssl

from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from datetime import datetime

import os

def send_email_report(filename, current_datetime, most_common_emotion, user_fullname, user_email):
    formatted_datetime = datetime.strptime(current_datetime, '%Y%m%d%H%M%S').strftime('%d/%m/%Y %H:%M:%S')

    subject = f"FER Application - Quick Report - {formatted_datetime}"
    body = f"Hi {user_fullname}!\nHere is the quick report for the video you uploaded.\nThe most common emotion detected is: {most_common_emotion}"
    sender_email = os.getenv("SENDER_EMAIL")
    receiver_email = user_email
    password = os.getenv("EMAIL_PASSWORD")

    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    #message["Bcc"] = receiver_email  # Recommended for mass emails

    # Add body to email
    message.attach(MIMEText(body, "plain"))

    #filename = "quick_report.pdf"  # In same directory as script

    # Open PDF file in binary mode
    with open(filename, "rb") as attachment:
        # Add file as application/octet-stream
        # Email client can usually download this automatically as attachment
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())

    # Encode file in ASCII characters to send by email    
    encoders.encode_base64(part)

    # Add header as key/value pair to attachment part
    part.add_header(
        "Content-Disposition",
        f"attachment; filename= {filename}",
    )

    # Add attachment to message and convert message to string
    message.attach(part)
    text = message.as_string()

    try:
        # Log in to server using secure context and send email
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, text)        

        print("Email sent successfully.")
        return True
    except Exception as e:
        print("Failed to send email.")
        return False

def send_email_full_report(filename, current_datetime, most_common_emotion, user_fullname, user_email):
    formatted_datetime = datetime.strptime(current_datetime, '%Y%m%d%H%M%S').strftime('%d/%m/%Y %H:%M:%S')

    subject = f"FER Application - Full Report - {formatted_datetime}"
    body = f"Hi {user_fullname}!\nHere is the full report for the video you uploaded.\nThe most common emotion detected is: {most_common_emotion}"
    sender_email = os.getenv("SENDER_EMAIL")
    receiver_email = user_email
    password = os.getenv("EMAIL_PASSWORD")

    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    #message["Bcc"] = receiver_email  # Recommended for mass emails

    # Add body to email
    message.attach(MIMEText(body, "plain"))

    #filename = "quick_report.pdf"  # In same directory as script

    # Open PDF file in binary mode
    with open(filename, "rb") as attachment:
        # Add file as application/octet-stream
        # Email client can usually download this automatically as attachment
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())

    # Encode file in ASCII characters to send by email    
    encoders.encode_base64(part)

    # Add header as key/value pair to attachment part
    part.add_header(
        "Content-Disposition",
        f"attachment; filename= {filename}",
    )

    # Add attachment to message and convert message to string
    message.attach(part)
    text = message.as_string()

    try:
        # Log in to server using secure context and send email
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, text)        

        print("Email sent successfully.")
        return True
    except Exception as e:
        print("Failed to send email.")
        return False