#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf
import os
from os.path import expanduser
import datetime
import time
import requests
import socket
from pynput.keyboard import Key, Listener
from threading import Thread


# Local Imports
import model, sample, encoder

#### Set desired model
DISERED_MODEL = "774M"



def usr_info(): # Function written to connect client over UDP/IP later if model is to weighty 
    datetime = time.ctime(time.time())
    user = expanduser("~")
    publicIP = requests.get('https://api.ipify.org/').text
    privateIP = socket.gethostbyname(socket.gethostname())
    key_buffer = f'[START OF LOGS]\n  *~ /Time: {datetime}\n  *~ User-Profile: {user}\n  *~ Public-IP: {publicIP}\n  *~ Private-IP: {privateIP}\n\n'
    print(key_buffer)
    return user


def keypress(Key):
    logged_data = []
    substitution = ['Key.enter', '[ENTER]\n', 'Key.backspace', '[BACKSPACE]', 'Key.space', ' ',
	'Key.alt_l', '[ALT]', 'Key.tab', '[TAB]', 'Key.delete', '[DEL]', 'Key.ctrl_l', '[CTRL]',
	'Key.left', '[LEFT ARROW]', 'Key.right', '[RIGHT ARROW]', 'Key.shift', '[SHIFT]', '\\x13', 
	'[CTRL-S]', '\\x17', '[CTRL-W]', 'Key.caps_lock', '[CAPS LK]', '\\x01', '[CTRL-A]', 'Key.cmd', 
	'[WINDOWS KEY]', 'Key.print_screen', '[PRNT SCR]', '\\x03', '[CTRL-C]', '\\x16', '[CTRL-V]']

    Key = (str(Key)).strip('\'')

    if Key in substitution:
        logged_data.append(substitution[substitution.index(Key)+1])
    else:
        logged_data.append(Key)
    return logged_data, Key

def run_logger(): # Threading a logger 
    with Listener(on_press = keypress) as listener:
        listener.join()

def join_logs(logged_data, key_buffer):
    print(key_buffer)  
    print(logged_data)

def predict_word(Key, logged_data): # WHen CNTRL-C is pressed should pass logged data to interact model
    if Key == '\\x01':
        interact_model(logged_data)
        print("predicting word")
    else:
        pass

def interact_model(
    text_input = [],
    model_name=DESIRED_MODEL,
    seed=None,
    nsamples=1,
    batch_size=None,
    length=1,
    temperature=0.5,
    top_k=40,
    top_p=1,
    models_dir='/home/{}/Desktop/AiSpellChecker/gpt-2/models'.format(user)
):

    """
    Interactively run the model
    :model_name=124M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
     :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
    """

    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        while True:
            raw_text = text_input
            while not raw_text:
                pass
            context_tokens = enc.encode(raw_text)
            generated = 0
            for _ in range(nsamples // batch_size):
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                })[:, len(context_tokens):]
                for i in range(batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                    print(text)
            print("=" * 80)


####Setupfunction
def setup(DESIRED_MODEL):

    def check_model_and_requirments(DESIRED_MODEL):
        models_dir = os.path.expanduser(os.path.expandvars(models_dir))
        if not models_dir:
            build_requirments()
            if DISERED_MODEL == "124M":
                download_model("124")
            if DISERED_MODEL == "355M":
                download_model("335M")
        if DISERED_MODEL == "774M":
                download_model("774M")
        else:
            print('Please modify veriable DESIRED_MODEL TO EITHER 124M, 335M, or 774M.')


    def build_requirments():
        FROM tensorflow/tensorflow:1.12.0-py3
        requirments = "fire>=0.1.3\nregex==2017.4.5\nrequests==2.21.0\ntqdm==4.31.1"

        os.system(mkdir /gpt-2)
        os.system(WORKDIR /gpt-2)
        os.system(ADD . /gpt-2)

            
    def download_model(model_string):
        import sys
        from tqdm import tqdm

        model = model_string

        subdir = os.path.join('models', model)
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        subdir = subdir.replace('\\','/') # needed for Windows

        for filename in ['checkpoint','encoder.json','hparams.json','model.ckpt.data-00000-of-00001', 'model.ckpt.index', 'model.ckpt.meta', 'vocab.bpe']:

            r = requests.get("https://storage.googleapis.com/gpt-2/" + subdir + "/" + filename, stream=True)

            with open(os.path.join(subdir, filename), 'wb') as f:
                file_size = int(r.headers["content-length"])
                chunk_size = 1000
                with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
                    # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        pbar.update(chunk_size)

    check_model_and_requirments(DESIRED_MODEL)


if __name__ == '__main__':
    setup(DESIRED_MODEL)

    fire.Fire(interact_model) # Start GPT2
    t1 = Thread(target=usr_info, args=()) # Grab and store user info TODO could probably store this on desk and not RAM
    t2 = Thread(target=run_logger, args=()) # Starts logged
    t3 = Thread(target=keypress, args=(Key)) # Starts key press substitation and formating. TODO store logs on RAM?
    t4 = Thread(target=predict_word, args=(Key, logged_data))
    

    
    t1.start()
    t2.start()
    t3.start()

    if Key:
        t4.start() # Only start process if a key is pressed. 
    else:
        pass

    t1.join()
    t1.join()
    t3.join()
    t4.join()

