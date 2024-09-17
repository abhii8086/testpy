import datetime

class Config:
    JWT_SECRET_KEY = 'sdhsahds8ieyrjkdjfd'
    JWT_ACCESS_TOKEN_EXPIRES = datetime.timedelta(days=1)
    MONGO_URI = "mongodb+srv://abhijeet:sharma45@geo-assist.tz3wp.mongodb.net/"  # MongoDB connection URI
