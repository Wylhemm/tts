import io
import re
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import torch
from TTS.api import TTS
from pydub import AudioSegment
from pydantic import BaseModel
import logging
import soundfile as sf

# Set up logging
logger = logging.getLogger("uvicorn")
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Pydantic model for request validation
class TTSRequest(BaseModel):
    text: str
    speaker_wav: str = 'download.mp3'
    language: str = 'es'

# Initialize the TTS model
device = "cuda" if torch.cuda.is_available() else "cpu"
tts_model_path = "tts_models/multilingual/multi-dataset/xtts_v2"
tts = TTS(tts_model_path).to(device)

def remove_emojis(text: str) -> str:
    """
    Remove emojis from the text by using a regular expression.
    """
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

@app.post('/clone_and_synthesize')
async def clone_and_synthesize(request: TTSRequest):
    try:
        # Remove emojis from text
        sanitized_text = remove_emojis(request.text)

        # Processing the TTS request
        wav = tts.tts(text=sanitized_text, speaker_wav=request.speaker_wav, language=request.language)
        logger.info(f"Generated waveform type: {type(wav)}")

        # Audio processing to MP3
        sample_rate = 24000  # High-quality sample rate
        wav_data = io.BytesIO()
        sf.write(wav_data, wav, sample_rate, format='WAV')
        wav_data.seek(0)
        audio_segment = AudioSegment.from_file(wav_data, format="wav")
        buffer = io.BytesIO()
        audio_segment.export(buffer, format="mp3")
        buffer.seek(0)

        return StreamingResponse(buffer, media_type='audio/mp3')

    except Exception as e:
        logger.error(f"Error in TTS processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5005)