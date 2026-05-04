# Preview what will be generated (no API calls)
# python3 generate_benchmark.py --dry-run

# # Generate 5 random variants to test
#python3 generate_benchmark.py --count 5

# # Generate the full grid (4 countries × 2 sexes × 5 panel sets × 5 styles = 200 variants)
# python3 generate_benchmark.py

# apt install -y \
#   libcairo2 \
#   libpango-1.0-0 \
#   libpangocairo-1.0-0 \
#   libgdk-pixbuf-2.0-0 \
#   libglib2.0-0 \
#   libffi-dev \
#   shared-mime-info

# printf "deb http://archive.ubuntu.com/ubuntu jammy main universe multiverse restricted\n" > /etc/apt/sources.list
# printf "deb http://archive.ubuntu.com/ubuntu jammy-updates main universe multiverse restricted\n" >> /etc/apt/sources.list
# printf "deb http://security.ubuntu.com/ubuntu jammy-security main universe multiverse restricted\n" >> /etc/apt/sources.list

python3 generate_benchmark_all.py --types mri --count 10