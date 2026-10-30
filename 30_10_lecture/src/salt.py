import hashlib

def get_student_params(student_id: str):
    student_id = str(student_id).strip()
    if not student_id.isdigit() or len(student_id) != 8:
        raise ValueError("Student ID must contain exactly 8 digits.")
    h = int(hashlib.sha256(student_id.encode("utf-8")).hexdigest()[:8], 16)
    n = 1024 + ((h % 9) * 64)
    return {
        "student_id": student_id,
        "N": n,
        "buffer_elements": 16_777_216,
        "seed": h % 100_007,
    }
