# -*- coding: UTF-8 -*-

from __future__ import print_function
import sys
import pika
import redis
import json
import ProblemDecoder_predict as pd
import os
import tensorflow as tf
import operator,time
import traceback
import numpy as np
import ConfigParser
import time
import argparse

def print(msg):
    sys.stdout.write(str(msg))
    sys.stdout.flush()




def make_float(nlist):
    return [float(x) for x in nlist]



def set_up(arg):
    rootpath = arg

    data_dir=os.path.join(rootpath,'data')
    problems = 'symptom_asking_problem'
    model = 'transformer'
    hparams_set = 'transformer_small'
    t2t_usr_dir=os.path.join(rootpath,'src')
    output_dir=os.path.join(rootpath,'model')
    hparams_key_value = "pos=none"

    problemD = pd.ProblemDecoder_predict(problem=problems,
                        model_dir=output_dir,
                        model_name=model,
                        hparams_set=hparams_set,
                        usr_dir=t2t_usr_dir,
                        data_dir=data_dir,
                        batch_size_specify=2,
                        return_beams=True,
                        beam_size=10,
                        write_beam_scores=True,
                        hparams_key_value=hparams_key_value,
                        eos_required=False,
                        extra_length=10
                        )

    return problemD



#argParser setup
parser=argparse.ArgumentParser(description='parse: ip_address, port, usr, password, rootpath')

parser.add_argument('-rabbit_ip',help='set rabbitmq ip_address, with default 192.168.1.242',type=str,default='192.168.1.242')
parser.add_argument('-rabbit_port',help='set rabbitmq port, with default 5672',type=int,default=5672)
parser.add_argument('-redis_ip',help='set redis ip_address, with default 192.168.1.242',type=str,default='192.168.1.242')
parser.add_argument('-redis_port',help='set redis port, with default 6379',type=int,default=6379)
parser.add_argument('-usr',help='set rabbitmq usr, with default rxthinking',type=str,default='rxthinking')
parser.add_argument('-password',help='set rabbitmq password, with default gniknihtxr',type=str,default='gniknihtxr')
parser.add_argument('-rootpath',help='set rootpath, with default ../',type=str,default='../')

pargs=parser.parse_args()



#设置rabbit和redis的连接
redis_addr=(pargs.redis_ip,pargs.redis_port)
rabbitmq_addr=(pargs.rabbit_ip,pargs.rabbit_port)

redisClient =redis.Redis(host=redis_addr[0], port=redis_addr[1])
cred = pika.credentials.PlainCredentials(pargs.usr,pargs.password)
connection = pika.BlockingConnection(pika.ConnectionParameters(host=rabbitmq_addr[0], port=rabbitmq_addr[1], credentials=cred))
#connection = pika.BlockingConnection(pika.ConnectionParameters(host=rabbitmq_addr[0], port=rabbitmq_addr[1]))
channel = connection.channel()



#模型初始化
rootpath=pargs.rootpath
setup = set_up(rootpath)



#listen to rabbit
def callback(ch, method, properties, body):
    print(" [x] Received %s\n" % body)

    try:
        #读取rabbitmq里面的信息
        astart=time.time()
        js = json.loads(body)
        requestId=js.get('requestId')
        sendin=[js['text'].encode(encoding='UTF-8',errors='strict')]



        start=time.time()
        #处理信息，得到结果和评分
        #temp_input='wrong input'
        result,score=setup.infer_singleSample(sendin[0],10)
        #result,score=setup.infer_singleSample(temp_input,10)
        score=make_float(score)
        end=time.time()
        print('problem decoder duration: %s\n' % (end-start))



        #处理result，score为json格式送入redis
        toRedis = {'code':0,'result':[]}
        existDict = []
        count = 0
        for x in result:
            tempDict = {'type': 'symptom', 'key': '', 'weight': 0}
            tempDict['weight'] = score[count]
            count = count + 1

            xstring = x[:x.find('<eos>')]
            xlist = xstring.split()
            for y in xlist:
                if y in existDict:
                    pass
                else:
                    tempDict['key'] = y
                    existDict.append(y)
                    break
            if tempDict['key'] == '':
                pass
            else:
                toRedis['result'].append(tempDict)



        ch.basic_ack(delivery_tag=method.delivery_tag)



        toRedisString=json.dumps(toRedis,ensure_ascii=False)
        redisClient.setex(requestId, toRedisString,1800)
        print('Send %s to redis with id %s\n' % (toRedisString,requestId))
        aend=time.time()
        print('total duration: %s\n' % (aend-astart))


    except:
        toRedis={'code':1,'message':traceback.format_exc()}
        print('Wrong input fromat, send back error message %s\n' % json.dumps(toRedis))
        print(traceback.format_exc())
        ch.basic_ack(delivery_tag = method.delivery_tag)
        toRedisString=json.dumps(toRedis)
        redisClient.setex(requestId, toRedisString,1800)
        return



channel.basic_consume(callback, queue='suggestion_Q')

print(' [*] Waiting for messages. To exit press CTRL+C\n')
print('rabbitmq ip: %s\n' % pargs.rabbit_ip)
print('rabbitmq port: %s\n' % pargs.rabbit_port)
print('redis ip: %s\n' % pargs.redis_ip)
print('redis port: %s\n' % pargs.redis_port)
print('rabbitmq usr: %s\n' % pargs.usr)
print('rabbitmq password: %s\n' % pargs.password)
print('rootpath: %s\n' % pargs.rootpath)

channel.start_consuming()
