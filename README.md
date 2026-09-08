# Interactive Voice Cloner MVP

This project is a Python CLI voice-cloning MVP. It uses the pretrained Coqui **XTTS-v2** model: the model derives a speaker embedding from your reference recording, uses it to condition text-to-speech, and writes a cloned WAV for every prompt you enter.

## Architecture and working

```text
authorised reference WAV
          |
          v
XTTS-v2 speaker encoder --> speaker embedding
                                |
text entered in the CLI --------+--> XTTS-v2 decoder --> outputs/cloned/*.wav
                                |
                           repeat until /quit
```

On startup, the CLI loads the pretrained model. For every `text>` prompt, it passes both the prompt and reference WAV to XTTS-v2; the backend extracts/caches speaker features and generates a new WAV in the reference speaker's style. It remains open for the next prompt until you enter `/quit`.

## Run it

The compatible Python 3.11 environment is `.venv-clone`.

```powershell
.\.venv-clone\Scripts\Activate.ps1
python voice_cloner_cli.py --reference path\to\your_voice.wav --language en
```

On first use, XTTS-v2 downloads its pretrained weights. Provide a clean 6-30 second WAV with one speaker and little background noise. Generated files are saved under `outputs/cloned/`.

Example:

```text
text> Hello, this sentence is generated in my cloned voice.
Generated synthetic audio: outputs\cloned\cloned_....wav
text> /quit
```

Use only a voice you own or have permission to use; generated files are marked as synthetic.
