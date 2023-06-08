# パッケージの導入
import RPi.GPIO as GPIO
import adafruit_dht
import board
from time import sleep
from time import time
import subprocess
import struct
import pyaudio
import pvporcupine
import paho.mqtt.client as mqtt
import json
import ift
from concurrent.futures import ThreadPoolExecutor

GPIO.setwarnings(False)

# LED出力、dht11の利用の設定
pinLED = 26
GPIO.setup(pinLED, GPIO.OUT)
dhtDevice = adafruit_dht.DHT11(board.D16)

# porcupineの生成
keywards = ["ok google"] # ウェイクワードを指定
accesskey = 'vfB8ODJ4KpruEmjaRshuXicwl6mZEdIHJTs+RSuy2Twm+tNWubRGPQ=='
porcupine = pvporcupine.create(accesskey,keywords=keywards)
print(keywards)

# オーディオストリームの生成
pa = pyaudio.PyAudio()
audio_stream = pa.open(
    rate=porcupine.sample_rate,
    channels=1,
    format=pyaudio.paInt16,
    input=True,
    frames_per_buffer=porcupine.frame_length)

# mptt（beebotte)のサブスクライブ設定
TOKEN = "token_oKmSdb8x6ZlC1yyG"
HOSTNAME = "mqtt.beebotte.com"
PORT = 8883
TOPIC = "MySmartHome/voice"
CACERT = "mqtt.beebotte.com.pem"

while True:
    # 音声認識
    pcm = audio_stream.read(porcupine.frame_length)
    pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
    result = porcupine.process(pcm)

    # ウェイクワード検出時の処理
    if result >= 0:
        subprocess.Popen(['~/env/bin/googlesamples-assistant-pushtotalk --lang ja-JP --project-id gapi-211003-6121f --device-model-id gapi-211003-model'],shell=True)
        GPIO.output(pinLED, GPIO.HIGH)
        sleep(0.3)
        GPIO.output(pinLED, GPIO.LOW)

        # 接続切断の処理
        def t():
            sleep(10)
            client.disconnect()

        pool = ThreadPoolExecutor(2)
        pool.submit(t)

        # mqtt接続成功時の処理
        def on_connect(client, userdata, flags, respons_code):
            print('status{0}'.format(respons_code))
            client.subscribe(TOPIC)

        # webリクエスト受信時の処理
        def on_message(client, userdata, msg):
            data=[]
            data = json.loads(msg.payload.decode("utf-8"))["data"]
            message = str(data[0]+data[1])

            say_r = "今は"
            fukai85 = "暑くてたまらない状態です。クーラーをつけるか、クーラーの温度を下げることをおすすめします。"
            fukai80 = "暑くて汗が出る状態です。クーラーをつけるか、クーラーの温度を下げることをおすすめします。"
            fukai75 = "少しだけ暑い状態です。"
            fukai70 = "暑くは無い状態です。"
            fukai65 = "非常に心地良い状態です。"
            fukai60 = "寒くはない状態です。"
            fukai55 = "肌寒い状態です。"
            fukai50 = "寒い状態です。暖房をつけるか、暖房の温度を上げることをおすすめします。"
            fukai45 = "非常に寒い状態です。暖房をつけるか、暖房の温度を上げることをおすすめします。"
            fukai40 = "寒くて堪らない状態です。暖房をつけるか、暖房の温度を上げることをおすすめします。"
            humi_low = "部屋が非常に乾燥しているので加湿器をつけることをお勧めします。"

            while True:
                try:
                    # 温度・湿度計測、テキスト生成、LED点灯
                    temp = dhtDevice.temperature
                    humi = dhtDevice.humidity
                    value = round(0.81*temp+0.01*humi*(0.99*temp-14.3)+46.3,1)
                    temp_say = str(temp)
                    humi_say = str(humi)
                    value_say = str(value)
                    say_t = "温度" + temp_say + "度、"
                    say_ts = "温度は" + temp_say + "度です。"
                    say_h = "湿度" + humi_say + "パーセント、"
                    say_hs = "湿度は" + humi_say + "パーセントです。"
                    say_v = "不快指数は" + value_say + "です。"

                    if value > 85:
                        say_f = fukai85
                    elif value > 80:
                        say_f = fukai80
                    elif value > 75:
                        say_f = fukai75
                    elif value > 70:
                        say_f = fukai70
                    elif value > 65:
                        say_f = fukai65
                    elif value > 60:
                        say_f = fukai60
                    elif value > 55:
                        say_f = fukai55
                    elif value > 50:
                        say_f = fukai50
                    elif value > 45:
                        say_f = fukai45
                    else:
                        say_f = fukai40

                    if humi < 40:
                        say_hl = humi_low
                        if value > 65 and value < 70:
                            say_head = "しかし、"
                        else:
                            say_head = "また、"
                    else:
                        say_hl = " "
                        say_head = " "

                    if message == "状態":
                        say = str(say_t + say_h + say_v + say_r + say_f + say_head + say_hl)
                    elif message == "温度":
                        say = str(say_ts)
                    elif message == "湿度":
                        say = str(say_hs)

                    GPIO.output(pinLED, GPIO.HIGH)
                    sleep(2)
                    GPIO.output(pinLED, GPIO.LOW)

                    # 音声出力
                    subprocess.run(['curl "https://api.voicetext.jp/v1/tts" -u "6z9r3e4h3fwlb2rn:" -d "text=' +say+ '" -d "speaker=hikari" -d "pitch=73" -d "speed=105" -d "volume=50" | play -'],shell=True)
                    print(say)

                    if message == "状態": # LINEに送信
                        ift.ifttt_webhook("notify",temp_say,humi_say,say_f+say_head+say_hl)

                    client.disconnect() #beebotteの接続切断処理
                    break

                except:
                    sleep(0) #瞬時にtryを再実行

        # mqttの設定
        client = mqtt.Client()
        client.username_pw_set("token:%s"%TOKEN)
        client.on_connect = on_connect
        client.on_message = on_message
        client.tls_set(CACERT)
        client.connect(HOSTNAME, port=PORT, keepalive=60)
        client.loop_forever()

GPIO.cleanup()
