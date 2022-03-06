export data_file='repo_text_files'

python train_main.py -e 20 -ma GAN -G 1 -de 20 -ge 10 -B 20 -lb 1 -lb_b 0.01 -sc FSE -df ${data_file}
python train_main.py -e 20 -ma GAN -G 1 -de 20 -ge 10 -B 20 -lb 1 -lb_b 0.01 -sc SE -df ${data_file}

python train_main.py -e 20 -ma GAN -G 1 -de 20 -ge 10 -B 20 -lb 1 -lb_b 0.1 -sc FSE -df ${data_file}
python train_main.py -e 20 -ma GAN -G 1 -de 20 -ge 10 -B 20 -lb 1 -lb_b 0.1 -sc SE -df ${data_file}