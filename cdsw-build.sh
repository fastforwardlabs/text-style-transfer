# Clone huggingface transformers from remote
git clone https://github.com/huggingface/transformers
cd ./transformers
git checkout tags/v4.18.0

# Install Huggingface Transformers from source (already cloned)
pip3 install .

# Install examples specific requirements
cd ./examples/pytorch/summarization && pip3 install -r requirements.txt

# Install TST project requirements
cd ~ && pip3 install -r requirements.txt