import requests

def ifttt_webhook(eventid,v1,v2,v3):
    payload = {"value1": v1+"度", "value2": v2+"％", "value3": v3}
    url = "https://maker.ifttt.com/trigger/" + eventid + "/with/key/h2iA9DjpwCVHtawXFKhENz13SXMdHeUqTiux8PKBsEK"
    response = requests.post(url,data=payload)
