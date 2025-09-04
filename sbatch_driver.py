import os

outdir = "/p/lustre2/nexouser/data/StanfordData/angelico/DBM/"

runs = [{
    "eta": 0.2,
    "triangle_theta": 30.0,
    "triangle_height": 100,
    "max_steps": 1000000,
}, {
    "eta": 0.6,
    "triangle_theta": 30.0,
    "triangle_height": 100,
    "max_steps": 1000000,
}, {
    "eta": 1,
    "triangle_theta": 30.0,
    "triangle_height": 100,
    "max_steps": 1000000,
}, {
    "eta": 1.4,
    "triangle_theta": 30.0,
    "triangle_height": 100,
    "max_steps": 1000000,
}, {
    "eta": 1.8,
    "triangle_theta": 30.0,
    "triangle_height": 100,
    "max_steps": 1000000,
}]


#form outputfile names that are clear for each setting
for i in range(len(runs)):
    x = runs[i]
    filetag = "eta_{:d}_theta_{:d}_height_{:d}.p".format(int(x['eta']*10), int(x['triangle_theta']), int(x['triangle_height']))
    runs[i]["filetag"] = filetag 
    runs[i]["jobname"] = f"dbm-{x['eta']}"


mem = 32768
activate_venv = 'source $HOME/my_personal_env/bin/activate'
for i in range(len(runs)):
    x = runs[i]
    fullpath = os.path.join(outdir, x['filetag'])
    cmd_options = '--export=ALL -p pbatch -t 32:00:00 -n 1 -J {} --mem-per-cpu={:d} -o {}.out'.format(x['jobname'], mem, fullpath[:-2])
    exe = 'python $HOME/DBM/runhpcscript.py {} {} {} {} {}'.format(x["eta"], x["triangle_theta"], x["triangle_height"], x["max_steps"], fullpath)
    cmd_full = '{} && sbatch {} --wrap=\'{}\''.format(activate_venv,cmd_options,exe)


    print(cmd_full)
    #os.system(cmd_full)
    print('job {} submitted'.format(x["jobname"]))


    

