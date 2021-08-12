import os


TOKEN = '1900850036:AAHinBkKM69bwZRVYdHqJMHQS3ePx2ZLSwg'
IP_ADDRESS = '20.106.188.252'
PORT = int(os.environ.get('PORT', '8443'))
# https://api.telegram.org/bot1900850036:AAHinBkKM69bwZRVYdHqJMHQS3ePx2ZLSwg/getWebhookInfo
# {"ok":true,"result":{"url":"https://20.106.188.252/1900850036:AAHinBkKM69bwZRVYdHqJMHQS3ePx2ZLSwg","has_custom_certificate":true,"pending_update_count":3,"last_error_date":1628255665,"last_error_message":"Wrong response from the webhook: 404 Not Found","max_connections":40,"ip_address":"20.106.188.252"}}


# openssl req -newkey rsa:2048 -sha256 -nodes -keyout private.key -x509 -days 365 -out cert.pem -subj "/C=US/ST=New York/L=Brooklyn/O=Example Brooklyn Company/CN=20.106.188.252"