import subprocess
import tempfile
import os
import re
import types

import numpy as np

def extract_flags(arg_dict, flag_list):
    raised_flags = []
    for flag in flag_list:
        try:
            if arg_dict[flag] == True:
                raised_flags.append(f'--{flag}')
            else:
                pass
        except KeyError:
            pass
        
    filtered_dict = {key: value for key, value in arg_dict.items() if key not in flag_list}
    return filtered_dict, raised_flags

def isint(x):
    # https://stackoverflow.com/a/15357477
    try:
        a = float(x)
        b = int(a)
    except (TypeError, ValueError):
        return False
    else:
        return a == b

def streme(p, n=None, order=None, kmer=None, bfile=None, objfun=None, 
           dna=None, rna=None, protein=None, alph=None, minw=None, maxw=None, 
           w=None, neval=None, nref=None, niter=None, thresh=None, evalue=None, patience=None, 
           nmotifs=None, time=None, totallength=None, hofract=None, seed=None, align=None, 
           desc=None, dfile=None):
    
    tmp = tempfile.NamedTemporaryFile(delete=False)
    
    try:
    
        if isinstance(p, str):
            streme_cmd = ['streme', '--p', p]
        else:
            streme_cmd = ['streme', '--p', tmp.name]
            
            if isinstance(p, list) or isinstance(p, types.GeneratorType):
                for i, seq in enumerate(p):
                    tmp.write(f'>seq_{i}\n'.encode('utf-8'))
                    tmp.write(f'{seq}\n'.encode('utf-8'))
                    
            elif isinstance(p, dict):
                for i, pack in enumerate(p.items()):
                    key, value = pack
                    if '>' != key[0]:
                        key = '>' + key
                    key += f'_{i}'
                    tmp.write(f'{key}\n'.encode('utf-8'))
                    tmp.write(f'{value}\n'.encode('utf-8'))
                    
            tmp.close()

        streme_args= {
            'n': n,
            'order': order,
            'kmer': kmer,
            'bfile': bfile,
            'objfun': objfun,
            'dna': dna,
            'rna': rna,
            'protein': protein,
            'alph': alph,
            'minw': minw,
            'maxw': maxw,
            'w': w,
            'neval': neval,
            'nref': nref,
            'niter': niter,
            'thresh': thresh,
            'evalue': evalue,
            'patience': patience,
            'nmotifs': nmotifs,
            'time': time,
            'totallength': totallength,
            'hofract': hofract,
            'seed': seed,
            'align': align,
            'desc': desc,
            'dfile': dfile
        }
        streme_args= {key: value for key, value in streme_args.items() if value is not None}
        streme_args, flags = extract_flags(streme_args, ['dna', 'rna', 'protein', 'evalue'])

        for key, value in streme_args.items():
            streme_cmd.append(f'--{key}')
            streme_cmd.append(str(value))

        streme_cmd += flags
        streme_cmd += ['--text']

        streme_process = subprocess.Popen(streme_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = streme_process.communicate()

    finally:
        tmp.close()
        os.unlink(tmp.name)
        
    return {'output':out, 'error':err}

def parse_streme_output(streme_output):
    streme_text = streme_output.decode("utf-8")

    # Metadata
    alphabet = re.search('ALPHABET= (.+)', streme_text)[1]
    alphabet = [ char for char in alphabet ]
    FREQ_scan = re.compile(r'Background letter frequencies\n(.+?)\n',re.DOTALL)
    freq_str = FREQ_scan.search(streme_text)[1]
    frequencies = {}
    for char in alphabet:
        frequencies[char] = float(re.search(f'{char} (.+?) ', freq_str)[1])
    metadata = {'alphabet': alphabet, 'frequencies': frequencies}
    
    
    # Motif Data
    results = []
    MEME_scan = re.compile(r'MOTIF (.*?)\n\n',re.DOTALL)
    motif_results = MEME_scan.findall(streme_text)
    for motif_data in motif_results:
        tag = motif_data.split('\n')[0].split('-')[1].replace(' STREME','')
        summary = re.findall('(\w+)= (\S+)', motif_data.split('\n')[1])
        summ_dict = {}
        for key, value in summary:
            if isint(value):
                try:
                    summ_dict[key] = int(value)
                except ValueError:
                    summ_dict[key] = int(float(value))
            else:
                summ_dict[key] = float(value)
        pwm = []
        for row in motif_data.split('\n')[2:]:
            pwm.append( [ float(val) for val in row.lstrip().rstrip().split() ])
        pwm = np.array(pwm).T
        
        results.append( {'tag': tag, 'summary': summ_dict, 'ppm': pwm} )
    return {'meta_data': metadata,'motif_results': results}
