name=PentaPeptideBackBoneSweepFinal

for gen_dim in 256 512; do
    for dis_dim in 256 512; do
        for latent in 128; do
            for gen_net in simple; do
                for dis_net in simple; do
                    for opt in adam rmsprop; do
                        version="gen=${gen_dim}_dis=${dis_dim}_latent=${latent}_genNet=${gen_net}_disNet=${dis_net}_opt=${opt}";
                        cp sbatch_template.sbatch sbatch.sbatch;

                        sed -i "s/NAME/$name/g" sbatch.sbatch;
                        sed -i "s/VERSION/$version/g" sbatch.sbatch;

                        args="--model.gen_hidden_dim=${gen_dim} --model.dis_hidden_dim=${dis_dim} --model.latent_dim=${latent} --model.gen_network_type=${gen_net} --model.dis_network_type=${dis_net} --model.opt=${opt}"
                        sed -i "s/ARGS/$args/g" sbatch.sbatch;
                        sbatch sbatch.sbatch;
                    done;
                done;
            done;
        done;
    done;
done
