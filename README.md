# Set up your own voice changer

There are two ways to run it:

- **One Windows PC (easiest):** the RVC Voice app. Needs an NVIDIA graphics card. See the next section.
- **Two machines:** a Mac or second PC runs the voice model so your gaming PC keeps its full FPS. See "Two machines" below.

## One Windows PC: the RVC Voice app

1. Download [**RVCVoice.exe**](https://github.com/sapandeep31/rvc-lan-voice/releases/download/app-v1.0.0/RVCVoice.exe) and open it.
   Windows shows "Windows protected your PC" because the app is not signed: click **More info**, then **Run anyway**, and allow it to make changes.
2. Pick your microphone and your headphones, then press **Install and start**. It downloads about 6 GB once
   (10 to 20 minutes) and installs everything itself, including the VB-Cable virtual audio cable. If Windows asks to install the driver, click **Install**.
3. When the app opens, press the round button to switch the voice changer on.
4. In Discord or your game, choose **CABLE Output (VB-Audio Virtual Cable)** as the microphone.

The app has a **Self check** tab (record 5 seconds and hear your new voice), a **Voice** tab with the sliders
described under "Tune the voice" below, and a **Devices** tab. More detail: [desktop/README.md](desktop/README.md).

## Two machines

Your mic goes from the PC into a **server** that runs the AI voice model, and the converted voice comes
back into the PC as a virtual microphone that Discord and games can use.

```
PC: mic ──► client ──(Wi-Fi/LAN)──► server (Mac or Windows) ──► client ──► "CABLE Input" ──► Discord / game
```

- **Server**: any Apple Silicon Mac (M1 or newer), or a Windows PC with an NVIDIA GPU.
- **Client**: a Windows PC with your mic. It uses almost no CPU or GPU.
- They can be the same Windows PC, but then the voice model uses that PC's GPU. A separate Mac keeps games at full FPS.

## 1. Get the code

On every machine you use:

```
git clone https://github.com/sapandeep31/rvc-lan-voice.git
cd rvc-lan-voice
```

## 2. Server

### On a Mac
1. Install [uv](https://docs.astral.sh/uv/): `brew install uv`
2. Install (once; downloads ~1 GB plus the voice models): `./setup/mac/install.sh`
3. Start: `./setup/mac/start.sh`. If port 8000 is taken, use `PORT=8010 ./setup/mac/start.sh`.
   The window prints the `server_ip`, `server_port` and the control-panel link. Keep it open.

### On Windows (NVIDIA GPU)
1. Install uv: in PowerShell, `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
2. Double-click `setup\windows\install_server.bat` (once; downloads ~3 GB).
3. Double-click `setup\windows\start_server.bat` and allow Python through the Windows Firewall.

## 3. Client (Windows PC with the mic)

1. Install [Python 3.12](https://www.python.org/downloads/) (tick "Add python.exe to PATH") and
   [VB-Audio Virtual Cable](https://vb-audio.com/Cable/), then restart.
2. Double-click `setup\windows\install_client.bat`.
3. Edit `config.json`, in the `"client"` block:
   - `"server_ip"`: what the server printed (`"127.0.0.1"` if the server is on this same PC)
   - `"server_port"`: same as the server (8000 unless you changed it)
   - `"input_device_keyword"`: part of your mic's name, e.g. `"TONOR"` or `"USB"`
4. Double-click `setup\windows\start_client.bat`.
   To hear yourself, pass your headphones' name: `start_client.bat "WH-CH720N"`.
5. In Discord or the game, choose **CABLE Output** as your microphone.

## 4. Tune the voice

Open the control panel on any phone or computer on the same Wi-Fi: `http://<server_ip>:<port>/panel`.
Every slider applies live, and all open panels stay in sync.

| Slider | What it does |
|---|---|
| Pitch | Semitones up. Male to female voice is usually 9 to 12. |
| Formant | Timbre. Lower sounds fuller and more mature, higher sounds younger. |
| Index ratio | How much it sounds like the model. Lower is clearer, especially for Hindi or other non-English speech. |
| Noise gate | Mutes background noise. Press **Calibrate to room noise** and stay quiet for 6 s. |
| Chunk | Audio per packet. Smaller means less delay; keep "Processing per chunk" green. |
| Context | Past audio the model hears. More gives clearer words, costs more processing, and adds no delay. |

The panel resets to the defaults when the server restarts. To make your values permanent, put them in
`config.json` under `"server"` (`default_pitch`, `default_formant`, `default_index_ratio`, `silent_threshold`,
`chunk_size`, `extra_convert_size`).

## Voices

| Voice | Where it comes from | Pick it with |
|---|---|---|
| Narrator (default) | Female storytelling voice, [MIT, by 7TearZZZ](https://huggingface.co/7TearZZZ/RVC2_Female_Storytelling_Narrating_700_Epoch_RVC2), downloaded by the install script | `"default_model": "narrator"` |
| Kafka | In the repo | `"default_model": "kafka"` |

Other RVC v2 voices: make a folder `models/<Name>/` with one `.safetensors` (or `.pth`) and optionally one
`.index`, set `"default_model": "<name>"`, then restart the server. Only use voices you have the rights to,
and never clone a real person without their consent.

## Troubleshooting

| Problem | Fix |
|---|---|
| Client says it can't connect | Check `server_ip` and `server_port`, open `http://<server_ip>:<port>/health` in a browser on the PC, and allow Python in the server's firewall. |
| Voice stutters or has gaps | Raise Chunk on the panel. Wi-Fi spikes are the usual cause, and a LAN cable on the server helps most. On a Mac, `sudo ifconfig awdl0 down` (AirDrop off until reboot) removes most Wi-Fi spikes. |
| Words sound smeared | Lower Index ratio, or raise Context. |
| Fan or keyboard comes through | Press **Calibrate to room noise** on the panel. |
| "Port already in use" | Start the server on another port (see above) and set the same `server_port` on the client. |
