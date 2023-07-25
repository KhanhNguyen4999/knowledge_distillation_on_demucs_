
echo "*************DOWNLOAD DATA*******************"
gdown 1rK45zFH7s1HdKXo95Z5fOJ0808BdMtse # clean train
gdown 1-6x3p7Em0gYD3BsS36qqjA_NvqqrK9kn # noisy train
gdown 1AjmSb9Gv9wz8OHalyPofA6PtJQRXs75z # clean test
gdown 15q6UXo8jU6tjJfIiqqRCWw5iPGvchrQS # noisy train

# Unzip noise
echo "*************UNZIP MODEL*******************"
unzip -q noisy_trainset_28spk_wav.zip?sequence=6
unzip -q noisy_testset_wav.zip

# Unzip clean
unzip -q clean_trainset_28spk_wav.zip?sequence=2
unzip -q clean_testset_wav.zip

echo "*************Split trainset and testset*******************"
mkdir -p train/clean
mkdir -p train/noisy

mkdir -p test/clean
mkdir -p test/noisy

mkdir -p valentini/train
mkdir -p valentini/test

echo "*************DOWNSAMPLE MODEL*******************"

sudo apt install sox
ls clean_testset_wav | xargs -I {} sox clean_testset_wav/{} -r 16000 -c 1 -b 16 test/clean/{}_16k.wav
ls noisy_testset_wav | xargs -I {} sox noisy_testset_wav/{} -r 16000 -c 1 -b 16 test/noisy/{}_16k.wav
ls clean_trainset_28spk_wav | xargs -I {} sox clean_trainset_28spk_wav/{} -r 16000 -c 1 -b 16 train/clean/{}_16k.wav
ls noisy_trainset_28spk_wav | xargs -I {} sox noisy_trainset_28spk_wav/{} -r 16000 -c 1 -b 16 train/noisy/{}_16k.wav

echo "*************CLONE CODE*******************"
git clone https://github.com/KhanhNguyen4999/knowledge_distillation_on_demucs_.git
cd knowledge_distillation_on_demucs_
pip install -r requirements_cuda.txt
pip install julius
pip install hydra_core==0.11.3 hydra_colorlog==0.1.4

echo "*************CREATE DATA*******************"
# train
python -m denoiser.audio /content/train/clean > /content/valentini/train/clean.json
python -m denoiser.audio /content/train/noisy > /content/valentini/train/noisy.json

# test
python -m denoiser.audio /content/test/clean > /content/valentini/test/clean.json
python -m denoiser.audio /content/test/noisy > /content/valentini/test/noisy.json

