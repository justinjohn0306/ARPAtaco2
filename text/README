TACOTRON2-CMUDICT-PATCH

Most Uberduck tacotron2 voices don't support ARPAbet. This patch was made to ease CMUDict training without modifying the original dataset.

HOW TO USE THIS PATCH IN TRAINING COLAB

- Upload the zip file to the Colab runtime
- After downloading tacotron2, unzip the contents to the text/ folder relative to tacotron2 repo
- Copy merged.dict.txt to parent directory (/content/tacotron2/)
- Python 3 might not recognize the new files so restart the runtime and run all cells in order until #3
- Edit step #3 to include a parameter named use_cmudict that will change hparams.text_cleaners, like this:

use_cmudict = True #@param {type:"boolean"}
if use_cmudict is True:
    hparams.text_cleaners = ["cmudict_cleaners", "english_cleaners"]
else:
    hparams.text_cleaners = ["english_cleaners"]

  This effectively gives you the choice to add ARPAbet support for your model.

- You're done. Have fun being able to type in ARPAbet sequences in synthesis after you're done training!

LEGAL

This code is licensed under the MIT license.