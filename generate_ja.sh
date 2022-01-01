
ja_file="${1:-"_exp.ja"}"
mkdir -p "plots"
rm -f "${ja_file}"

# Set up your default options here
run="python train.py"
defaults="--cuda -s 1 -w 1 --log_scale"
default_run="${run} ${defaults}"

# Define the varying options here
REG_POWS=(2 3 4)
REG_COEFFS=(0.00001 0.0001 0.001 0.01 0.1 1.0 10.0)

# First, add the default run to job array
echo "${default_run} --save_fig 'plots/plot(default).png'" >> "${ja_file}"

# Then add all combinations of options
for reg_pow in "${REG_POWS[@]}"; do
    for reg_coeff in "${REG_COEFFS[@]}"; do
        command="${default_run}"
        command+=" --reg_pow ${reg_pow} --reg_coeff ${reg_coeff}"
        command+=" --save_fig 'plots/plot(p=${reg_pow}, c=${reg_coeff}).png'"
        echo "${command}" >> "${ja_file}"
    done
done

# Check job array file and number of jobs
echo "job array has $(wc -l < "${ja_file}") jobs:"
cat "${ja_file}"