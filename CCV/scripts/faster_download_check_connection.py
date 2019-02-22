#from __future__ import print_function
import argparse
import glob
import json
import os
import shutil
import subprocess
import uuid
from collections import OrderedDict

from joblib import delayed
from joblib import Parallel
import pandas as pd



import time
import datetime

try:
    import httplib
except:
    import http.client as httplib



def create_video_folders(dataset, output_dir, tmp_dir):
    """Creates a directory for each label name in the dataset."""
    if 'label-name' not in dataset.columns:
        this_dir = os.path.join(output_dir, 'test')
        if not os.path.exists(this_dir):
            os.makedirs(this_dir)
        # I should return a dict but ...
        return this_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    label_to_dir = {}
    for label_name in dataset['label-name'].unique():
        this_dir = os.path.join(output_dir, label_name)
        if not os.path.exists(this_dir):
            os.makedirs(this_dir)
        label_to_dir[label_name] = this_dir
    return label_to_dir


def construct_video_filename(row, label_to_dir, trim_format='%06d'):
    """Given a dataset row, this function constructs the
       output filename for a given video.
    """
    basename = '%s_%s_%s.mp4' % (row['video-id'],
                                 trim_format % row['start-time'],
                                 trim_format % row['end-time'])
    if not isinstance(label_to_dir, dict):
        dirname = label_to_dir
    else:
        dirname = label_to_dir[row['label-name']]
    output_filename = os.path.join(dirname, basename)
    return output_filename

def have_internet(host="www.google.com"):
    conn = httplib.HTTPConnection(host, timeout=5)
    try:
        conn.request("HEAD", "/")
        conn.close()
        return True
    except:
        conn.close()
        return False

def download_clip(video_identifier, output_filename,
                  start_time, end_time,
                  tmp_dir='/tmp/kinetics',
                  num_attempts=2,
                  url_base='https://www.youtube.com/watch?v='):
    """Download a video from youtube if exists and is not blocked.

    arguments:
    ---------
    video_identifier: str
        Unique YouTube video identifier (11 characters)
    output_filename: str
        File path where the video will be stored.
    start_time: float
        Indicates the begining time in seconds from where the video
        will be trimmed.
    end_time: float
        Indicates the ending time in seconds of the trimmed video.
    """
    # Defensive argument checking.
    assert isinstance(video_identifier, str), 'video_identifier must be string'
    assert isinstance(output_filename, str), 'output_filename must be string'
    assert len(video_identifier) == 11, 'video_identifier must have length 11'

    st = time.time()

    status = False 
    # Construct command line for getting the direct video link.
    tmp_filename = os.path.join(tmp_dir,
                                '%s.%%(ext)s' % uuid.uuid4())
    '''
    command = ['youtube-dl',
               '--quiet', '--no-warnings',
               '-f', 'mp4',
               '-o', '"%s"' % tmp_filename,
               '"%s"' % (url_base + video_identifier)]
    '''
    #USE -f 22 for the best quality

    #this is a faster version for dowloading Kinetics dataset which is more than 3 times faster than the initial version for one thread and more than 7 times faster multithread
    # tested for dowloading 10 videos (with one thread): 149.759222031 vs  453.865912914 
    #tested for dowloading 10 videos (with one multiple threads ):  44.76192379 vs 314.399228811
    command = ['ffmpeg',
                '-ss', str(start_time),
                '-t', str(end_time - start_time),
                '-i', '$(youtube-dl --socket-timeout 120 -f mp4 -g', '"%s"' % (url_base + video_identifier), ')',
                '-c:v', 'libx264', '-c:a', 'copy',
                '-threads', '1',
                '-strict', '-2',
                '-loglevel', 'panic',
                '"%s"' % output_filename]
                

    command = ' '.join(command)
    #print(command)
    wait_time_seconds = 2
    while True:
        if have_internet():
            attempts = 0
            while True:
                try:
                    print("Downloading video: %s. Time now: %s " %(output_filename,datetime.datetime.now()))
                    output = subprocess.check_output(command, shell=True,
                                                     stderr=subprocess.STDOUT)
                    #print("after subprocess.check_output")
                except (subprocess.CalledProcessError) as err:
                    attempts += 1
                    #print("[%s; %s; %s;]" %(status,output_filename, err.output))
                    print('Attempts download:', attempts, status, output_filename, err.output)
                    print('Time now: %s; sec passed: %s' %(datetime.datetime.now(), time.time() - st))
                    if os.path.exists(output_filename):
                        print("Deleting possible corrupted file!!!!!!!!!!!!!!!!!!: ", output_filename)
                        os.remove(output_filename)        
                    
                    if (attempts == num_attempts):
                        if  have_internet():
                            print('Not possible to download!! \n\n')
                            return status, err.output
                        else:
                            break
                else:
                    break
            
            if (not have_internet()):
                #print('continue')
                continue
            else:
                #print('break')
                break
        
        else:
            print("No Internet connection! time now: %s. Trying again after %.2f seconds" % (datetime.datetime.now(),wait_time_seconds))
            time.sleep(wait_time_seconds)

    '''
    tmp_filename = glob.glob('%s*' % tmp_filename.split('.')[0])[0]
    # Construct command to trim the videos (ffmpeg required).
    command = ['ffmpeg',
               '-i', '"%s"' % tmp_filename,
               '-ss', str(start_time),
               ''-c:v', 'libx264', '-c:a', 'copy',
               '-threads', '1',-t', str(end_time - start_time),
               
               '-loglevel', 'panic',
               '"%s"' % output_filename]
    command = ' '.join(command)
    try:
        output = subprocess.check_output(command, shell=True,
                                         stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        return status, err.output
    '''

    # Check if the video was successfully saved.
    status = os.path.exists(output_filename)
    #print(status)
    if status:
        try:
            command = 'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 ' + '"%s"'%output_filename
            duration = float(subprocess.check_output(command, shell=True,
                                     stderr=subprocess.STDOUT))
            if duration < 0.8:
                raise Exception

            print("Saved video: %s. Time now: %s; sec passed: %s, Duration Video %.2f \n" %(output_filename, datetime.datetime.now(), time.time() - st, duration))
            return status, 'Downloaded'
        except Exception as e:
                    print('!!!!!The video exists but it may by corrupted!!! ', output_filename, e.__doc__, e.message)
                    os.remove(output_filename)
                    print('Deleted the corrupted video: ', output_filename)
                    print('')
                    
                    return False, 'NOT Downloaded, corrupted'

        
        #print('Time now: %s; sec passed: %s' %(datetime.datetime.now(), time.time() - st))
        #print('')
    else:
        print("NOT saved video: %s!!!!!!!!!!!!!!!!!  Time now: %s; sec passed: %s \n" % (output_filename, datetime.datetime.now(), time.time() - st))
        return status, 'NOT Downloaded'
        #print('Time now: %s; sec passed: %s' %(datetime.datetime.now(), time.time() - st))
        #print('')

    #os.remove(tmp_filename)
    


def download_clip_wrapper(row, label_to_dir, trim_format, tmp_dir):
    """Wrapper for parallel processing purposes."""
    start_time = time.time()
    output_filename = construct_video_filename(row, label_to_dir,
                                               trim_format)
    clip_id = os.path.basename(output_filename).split('.mp4')[0]


    if os.path.exists(output_filename):

        try:
            command = 'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 ' + '"%s"'%output_filename
            #print(command)
            duration = float(subprocess.check_output(command, shell=True,
                                             stderr=subprocess.STDOUT))
            print('Duration video: ', output_filename, duration)

            if duration < 0.8:
                raise Exception



            if (duration > 7) and (duration < 16):
                status = tuple([clip_id, True, 'Exists'])
                print('The video exists already: %s. Time now: %s; sec passed: %s' %(output_filename, datetime.datetime.now(), time.time() - start_time))
                print('')
                return status
            else:
                #be sure that you don't delete a video that maybe is not available anymore on Youtube

                tmp_output_file = tmp_dir + '/' + clip_id + '.mp4'
                if os.path.exists(tmp_output_file):
                    os.remove(tmp_output_file)

                downloaded, log = download_clip(row['video-id'], tmp_output_file,
                                    row['start-time'], row['end-time'],
                                    tmp_dir=tmp_dir)
                try:
                    command = 'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 ' + '"%s"'%tmp_output_file
                    print(command)
                    tmp_duration = float(subprocess.check_output(command, shell=True,
                                             stderr=subprocess.STDOUT))
                    if tmp_duration < 0.8:
                        raise Exception

                    if abs(duration - 10) > abs(tmp_duration -10):
                        command = 'mv ' + '"%s"'%tmp_output_file + ' ' + '"%s"'%output_filename
                        print(command)
                        o = subprocess.check_output(command, shell=True,
                                             stderr=subprocess.STDOUT)
                        #print(command) #!!!!!!!!!!!!!!!1
                        print('Updated the video ', output_filename, duration,  tmp_duration)
                        print('')
                    else:
                        print('NOT updated the video ', output_filename,duration,  tmp_duration)
                        print('')
                        os.remove(tmp_output_file)

                    status = tuple([clip_id, True, 'Exists'])
                    return status

                except Exception as e:
                    print('The TMP video exists but it may by corrupted!!! ', e.__doc__, e.message)
                    os.remove(tmp_output_file)
                    print('Deleted tmp video: and keep the original', tmp_output_file)
                    print('')
                    status = tuple([clip_id, True, 'Exists'])
                    return status



                #os.remove(output_filename)
                #print('Deleted video: ', output_filename)

        except Exception as e:
            print('The video exists but it may by corrupted!!! ', e.__doc__, e.message)
            os.remove(output_filename)
            print('Deleted video: ', output_filename)
            pass

    
    downloaded, log = download_clip(row['video-id'], output_filename,
                                    row['start-time'], row['end-time'],
                                    tmp_dir=tmp_dir)
    status = tuple([clip_id, downloaded, log])
    #print('Time now: %s; sec passed: %s' %(datetime.datetime.now(), time.time() - start_time))
    return status


def parse_kinetics_annotations(input_csv, ignore_is_cc=False):
    """Returns a parsed DataFrame.

    arguments:
    ---------
    input_csv: str
        Path to CSV file containing the following columns:
          'YouTube Identifier,Start time,End time,Class label'

    returns:
    -------
    dataset: DataFrame
        Pandas with the following columns:
            'video-id', 'start-time', 'end-time', 'label-name'
    """
    df = pd.read_csv(input_csv)
    if 'youtube_id' in df.columns:
        columns = OrderedDict([
            ('youtube_id', 'video-id'),
            ('time_start', 'start-time'),
            ('time_end', 'end-time'),
            ('label', 'label-name')])
        df.rename(columns=columns, inplace=True)
        if ignore_is_cc:
            df = df.loc[:, df.columns.tolist()[:-1]]
    return df


def main(input_csv, output_dir,
         trim_format='%06d', num_jobs=24, tmp_dir='/tmp/kinetics',
         drop_duplicates=False, download_report='download_report.json'):

    # Reading and parsing Kinetics.
    dataset = parse_kinetics_annotations(input_csv)
    # if os.path.isfile(drop_duplicates):
    #     print('Attempt to remove duplicates')
    #     old_dataset = parse_kinetics_annotations(drop_duplicates,
    #                                              ignore_is_cc=True)
    #     df = pd.concat([dataset, old_dataset], axis=0, ignore_index=True)
    #     df.drop_duplicates(inplace=True, keep=False)
    #     print(dataset.shape, old_dataset.shape)
    #     dataset = df
    #     print(dataset.shape)

    # Creates folders where videos will be saved later.
    label_to_dir = create_video_folders(dataset, output_dir, tmp_dir)

    start_time = time.time()

    print(dataset.shape)

    #dataset = dataset.iloc[:60, :]
    #print(dataset.shape)
    #print(dataset)

    # Download all clips.
    if num_jobs == 1:
        status_lst = []
        for i, row in dataset.iterrows():
            status_lst.append(download_clip_wrapper(row, label_to_dir,
                                                    trim_format, tmp_dir))
    else:
        status_lst = Parallel(n_jobs=num_jobs)(delayed(download_clip_wrapper)(
            row, label_to_dir,
            trim_format, tmp_dir) for i, row in dataset.iterrows())

    print("--- Number of seconds to download video clips:  %s  ---" % (time.time() - start_time))

    # Clean tmp dir.
    shutil.rmtree(tmp_dir)

    # Save download report.
    print("Saving the download report: ", download_report)
    with open(download_report, 'w') as fobj:
        fobj.write(json.dumps(status_lst))


if __name__ == '__main__':
    description = 'Helper script for downloading and trimming kinetics videos.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('input_csv', type=str,
                   help=('CSV file containing the following format: '
                         'YouTube Identifier,Start time,End time,Class label'))
    p.add_argument('output_dir', type=str,
                   help='Output directory where videos will be saved.')
    p.add_argument('-f', '--trim-format', type=str, default='%06d',
                   help=('This will be the format for the '
                         'filename of trimmed videos: '
                         'videoid_%0xd(start_time)_%0xd(end_time).mp4'))
    p.add_argument('-n', '--num-jobs', type=int, default=24)
    p.add_argument('-t', '--tmp-dir', type=str, default='/tmp/kinetics')
    p.add_argument('--drop-duplicates', type=str, default='non-existent',
                   help='Unavailable at the moment')
                   # help='CSV file of the previous version of Kinetics.')
    p.add_argument('--download_report', type=str, default='download_report.json',
                   help='The file name (possibly with full path) whre to save the download report')
    main(**vars(p.parse_args()))
