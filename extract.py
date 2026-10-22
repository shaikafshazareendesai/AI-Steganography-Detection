"""
LSB Steganography Encoding and Decoding Functions
This module implements Least Significant Bit (LSB) steganography techniques
for hiding and extracting messages in images.
"""

import cv2
import numpy as np
import re
from typing import Tuple, Optional

MAGIC = b"STG1"


def _looks_like_meaningful_raw_text(decoded_text: str) -> bool:
    """Guardrail for unverified raw fallback to reduce random gibberish outputs."""
    if not decoded_text:
        return False

    text = decoded_text.replace("\x00", "").strip()
    if len(text) < 6:
        return False

    printable_ratio = sum(1 for ch in text if ch.isprintable()) / max(len(text), 1)
    alpha_ratio = sum(1 for ch in text if ch.isalpha()) / max(len(text), 1)
    has_word_separator = any(ch.isspace() for ch in text)
    
    # Additional checks to filter out binary garbage
    has_control_chars = any(ord(ch) < 32 and ch not in ('\t', '\n', '\r') for ch in text)
    has_extended_ascii = any(ord(ch) > 127 for ch in text)
    too_many_special_chars = sum(1 for ch in text if not ch.isalnum() and not ch.isspace()) / max(len(text), 1) > 0.3

    return (printable_ratio >= 0.95 and 
            alpha_ratio >= 0.5 and 
            has_word_separator and 
            not has_control_chars and 
            not has_extended_ascii and 
            not too_many_special_chars)


def _extract_meaningful_text_run(decoded_text: str) -> Optional[str]:
    """Return the best printable text run from a noisy decoded string, if one exists."""
    if not decoded_text:
        return None

    candidates = re.findall(r"[ -~]{6,}", decoded_text)
    best_candidate = None

    for candidate in candidates:
        text = candidate.replace("\x00", "").strip()
        if _looks_like_meaningful_raw_text(text):
            if best_candidate is None or len(text) > len(best_candidate):
                best_candidate = text

    return best_candidate


def _decode_unverified_raw_payload(binary_message: str) -> Optional[str]:
    """Try conservative legacy/raw decoding strategies for older images."""
    for bit_offset in range(8):
        raw_bytes = bits_to_bytes(binary_message[bit_offset:])
        if not raw_bytes:
            continue

        candidates = []

        prefix = raw_bytes.split(b"\x00", 1)[0]
        if prefix:
            candidates.append(prefix)

        candidates.append(raw_bytes)

        for candidate_bytes in candidates:
            decoded_text = candidate_bytes.decode("utf-8", errors="ignore").strip()
            if _looks_like_meaningful_raw_text(decoded_text):
                return decoded_text

            text_run = _extract_meaningful_text_run(decoded_text)
            if text_run:
                return text_run

    return None

def message_to_binary(message: str) -> str:
    """
    Convert a string message to binary representation.
    
    Args:
        message (str): The message to convert
        
    Returns:
        str: Binary representation of the message
    """
    if type(message) == str:
        return ''.join([format(ord(i), "08b") for i in message])
    elif type(message) == bytes or type(message) == np.ndarray:
        return [format(i, "08b") for i in message]
    elif type(message) == int or type(message) == np.uint8:
        return format(message, "08b")
    else:
        raise TypeError("Input type not supported")

def binary_to_message(binary_message: str) -> str:
    """
    Convert binary message back to string.
    
    Args:
        binary_message (str): Binary representation of message
        
    Returns:
        str: Decoded message
    """
    message = ""
    for i in range(0, len(binary_message), 8):
        byte = binary_message[i:i+8]
        message += chr(int(byte, 2))
    return message


def bits_to_bytes(binary_message: str) -> bytes:
    """Convert a bit-string to bytes (truncates trailing incomplete byte)."""
    usable = (len(binary_message) // 8) * 8
    if usable == 0:
        return b""
    return bytes(int(binary_message[i:i+8], 2) for i in range(0, usable, 8))


def _extract_bitstream(image: np.ndarray, channel_order: Tuple[int, int, int]) -> str:
    """Extract LSB bitstream from image with explicit channel scan order."""
    bits = []
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            for channel in channel_order:
                bits.append(str(int(image[row, col, channel] & 1)))
    return ''.join(bits)


def _decode_from_bitstream(binary_message: str, allow_unverified_raw: bool) -> Optional[str]:
    """Decode message from a provided bitstream using verified formats first."""
    # Preferred format: MAGIC + length + payload.
    raw = bits_to_bytes(binary_message)
    if len(raw) >= 8 and raw[:4] == MAGIC:
        message_len = int.from_bytes(raw[4:8], byteorder="big", signed=False)
        end = 8 + message_len
        if 0 <= message_len <= (len(raw) - 8):
            payload = raw[8:end]
            return payload.decode("utf-8", errors="replace")

    # Backward compatibility: delimiter-based legacy format ("###").
    delimiter_bits = "001000110010001100100011"  # "###" in binary
    idx = binary_message.find(delimiter_bits)
    if idx != -1:
        message_binary = binary_message[:idx]
        return binary_to_message(message_binary)

    if allow_unverified_raw:
        # Optional legacy fallback: raw UTF-8 bits followed by zero padding (no header/delimiter).
        # Also scan byte offsets so older build layouts can still be recovered.
        raw_candidate = _decode_unverified_raw_payload(binary_message)
        if raw_candidate:
            return raw_candidate

        raw_bytes = bits_to_bytes(binary_message)
        if raw_bytes:
            decoded = raw_bytes.decode("utf-8", errors="ignore").strip()
            text_run = _extract_meaningful_text_run(decoded)
            if text_run:
                return text_run

    return None

def hide_message(image_path: str, message: str, output_path: str) -> bool:
    """
    Hide a message in an image using LSB steganography.
    
    Args:
        image_path (str): Path to the input image
        message (str): Message to hide
        output_path (str): Path to save the stego image
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not read the image")

        # Encode as: MAGIC(4 bytes) + message_length(4 bytes) + utf-8 payload
        payload = message.encode("utf-8")
        header = MAGIC + len(payload).to_bytes(4, byteorder="big", signed=False)
        packet = header + payload

        # Convert packet bytes to binary
        binary_message = ''.join(format(b, '08b') for b in packet)
        
        # Calculate maximum payload size (bits capacity)
        max_payload_bits = image.shape[0] * image.shape[1] * 3
        
        if len(binary_message) > max_payload_bits:
            raise ValueError("Message too long to hide in this image")
        
        # Hide the message in the image
        message_index = 0
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                for color_channel in range(3):
                    if message_index < len(binary_message):
                        # Get the LSB of the pixel value
                        pixel_value = image[row, col, color_channel]
                        
                        # Replace LSB with message bit
                        lsb = pixel_value & 0xFE  # Clear LSB
                        message_bit = int(binary_message[message_index])
                        image[row, col, color_channel] = lsb | message_bit
                        
                        message_index += 1
                    else:
                        break
                if message_index >= len(binary_message):
                    break
            if message_index >= len(binary_message):
                break
        
        # Save the stego image
        cv2.imwrite(output_path, image)
        return True
        
    except Exception as e:
        print(f"Error hiding message: {e}")
        return False

def extract_message(image_path: str, allow_unverified_raw: bool = False) -> Optional[str]:
    """
    Extract a hidden message from a stego image using LSB steganography.
    
    Args:
        image_path (str): Path to the stego image
        
    Returns:
        Optional[str]: Extracted message if found, None otherwise
    """
    try:
        # Read the stego image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not read the image")

        # Try primary cv2 channel scan order (BGR), then legacy RGB order for older generated samples.
        for channel_order in ((0, 1, 2), (2, 1, 0)):
            binary_message = _extract_bitstream(image, channel_order)
            decoded = _decode_from_bitstream(binary_message, allow_unverified_raw=allow_unverified_raw)
            if decoded:
                return decoded

        return None
            
    except Exception as e:
        print(f"Error extracting message: {e}")
        return None

def test_lsb_steganography():
    """
    Test function to demonstrate LSB steganography.
    """
    # Create a simple test image
    test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    cv2.imwrite("test_image.png", test_image)
    
    # Test message
    test_message = "This is a secret hidden message!"
    
    # Hide the message
    success = hide_message("test_image.png", test_message, "stego_image.png")
    
    if success:
        print("Message hidden successfully!")
        
        # Extract the message
        extracted_message = extract_message("stego_image.png")
        
        if extracted_message:
            print(f"Extracted message: {extracted_message}")
            return True
        else:
            print("Failed to extract message")
            return False
    else:
        print("Failed to hide message")
        return False

if __name__ == "__main__":
    # Run test
    test_lsb_steganography()
