import os
import pdb

url_base='https://www.youtube.com/watch?v='

missing_files = []

a = file('testVidID.txt').readlines()#[4400:]#4658
for idx, line in enumerate(a):
	line = line.strip()
	print "Downloading {}, {}/{}".format(line+'.mp4', idx, len(a))
	if os.path.exists('./test/'+line+'.mp4') == True:
		continue

	command = ['ffmpeg',
           ' -i', ' $(youtube-dl --socket-timeout 120 -f mp4 -g', ' "%s"' % (url_base + line), ')',
           ' -c:v', ' libx264', ' -c:a', ' copy',
           ' -threads', ' 1',
           ' -strict', ' -2',
           ' -loglevel', ' panic',
           ' "%s"' % './test/'+line+'.mp4']

	if os.system("".join(command)) == 256:
		missing_files.append(line)
	#pdb.set_trace()

print missing_files
