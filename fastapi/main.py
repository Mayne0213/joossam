from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
import json

import cv2
import numpy as np

app = FastAPI(
    title="Joossam OMR Grading API",
    description="OMR 채점을 위한 FastAPI 서비스",
    version="1.0.0"
)

# CORS 설정 - Vercel에서 호스팅되는 프론트엔드 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://joossameng.vercel.app",
        "https://*.vercel.app",
        "http://localhost:3000",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 상수 및 설정 ---
TARGET_WIDTH = 2480
TARGET_HEIGHT = 3508


# --- 함수 정의 ---

def load_image_from_bytes(image_bytes: bytes):
    """바이트 데이터에서 이미지 로드"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지를 읽을 수 없습니다")
    return img, img.shape[:2]


def resize_image_to_target(img, target_width=TARGET_WIDTH, target_height=TARGET_HEIGHT):
    """이미지를 타겟 크기로 리사이징 (비율 유지)"""
    h, w = img.shape[:2]

    if w == target_width and h == target_height:
        return img, 1.0, 1.0

    scale_x = target_width / w
    scale_y = target_height / h

    new_width = int(w * scale_x)
    new_height = int(h * scale_y)

    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    return resized_img, scale_x, scale_y


def gamma_correction(img, gamma=0.7):
    """감마 보정으로 밝기 곡선 조정"""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)


def unsharp_mask(img, kernel_size=(5, 5), sigma=1.0, amount=1.0):
    """언샤프 마스킹으로 경계 선명화"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(img, kernel_size, sigma)
    sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    return sharpened


def deskew_image_with_barcodes(img):
    """바코드를 기준으로 이미지 기울기 보정"""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        img_height, img_width = img.shape[:2]
        top_area_threshold = img_height * 0.15
        
        top_rectangles = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if (y < top_area_threshold and
                w > 10 and h > 10 and
                w < 300 and h < 300):
                center_x = x + w // 2
                center_y = y + h // 2
                top_rectangles.append({'center': (center_x, center_y)})
        
        if len(top_rectangles) < 15:
            return img
        
        top_rectangles.sort(key=lambda x: x['center'][1])
        top_rectangles = top_rectangles[:23]
        top_rectangles.sort(key=lambda x: x['center'][0])
        
        if len(top_rectangles) >= 10:
            left_points = top_rectangles[:5]
            right_points = top_rectangles[-5:]
            
            left_avg_x = np.mean([p['center'][0] for p in left_points])
            left_avg_y = np.mean([p['center'][1] for p in left_points])
            right_avg_x = np.mean([p['center'][0] for p in right_points])
            right_avg_y = np.mean([p['center'][1] for p in right_points])
            
            delta_y = right_avg_y - left_avg_y
            delta_x = right_avg_x - left_avg_x
            
            if delta_x == 0:
                return img
            
            angle_rad = np.arctan2(delta_y, delta_x)
            angle_deg = np.degrees(angle_rad)
            
            if abs(angle_deg) < 0.3:
                return img
            
            if abs(angle_deg) > 10:
                return img
            
            center = (img_width // 2, img_height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
            
            deskewed = cv2.warpAffine(
                img, rotation_matrix, (img_width, img_height),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            
            return deskewed
        
        return img
        
    except Exception:
        return img


def preprocess_omr_image(img):
    """OMR 이미지 전처리 강화"""
    denoised = cv2.GaussianBlur(img, (3, 3), 0)
    gamma_corrected = gamma_correction(denoised, gamma=0.7)

    if len(gamma_corrected.shape) == 3:
        gray = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2GRAY)
    else:
        gray = gamma_corrected

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    sharpened = unsharp_mask(enhanced, amount=0.8)
    result = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    return result


def calculate_marking_density(img, x, y, width=30, height=60):
    """특정 좌표 주변 영역의 마킹 밀도 계산"""
    h, w = img.shape[:2]
    x1 = max(0, x - width//2)
    y1 = max(0, y - height//2)
    x2 = min(w, x + width//2)
    y2 = min(h, y + height//2)

    region = img[y1:y2, x1:x2]
    if region.size == 0:
        return 0.0

    if len(region.shape) == 3:
        region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    avg_darkness = 255 - np.mean(region)
    dark_pixels = np.sum(region < 180)
    total_pixels = region.size
    dark_ratio = dark_pixels / total_pixels
    medium_dark_pixels = np.sum(region < 160)
    medium_dark_ratio = medium_dark_pixels / total_pixels
    very_dark_pixels = np.sum(region < 120)
    very_dark_ratio = very_dark_pixels / total_pixels
    density_score = (avg_darkness / 255.0) * 0.2 + dark_ratio * 0.2 + medium_dark_ratio * 0.4 + very_dark_ratio * 0.2

    return density_score


upperValueSquare = 180


def find_top_black_rectangles(img):
    """상단 검은색 사각형들을 찾아서 좌표 반환"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, upperValueSquare, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    top_rectangles = []
    img_height = img.shape[0]
    top_area_threshold = img_height * 0.15

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (y < top_area_threshold and
            w > 10 and h > 10 and
            w < 300 and h < 300):
            center_x = x + w // 2
            center_y = y + h // 2
            top_rectangles.append({
                'center': (center_x, center_y),
                'bbox': (x, y, w, h),
                'area': w * h
            })

    top_rectangles.sort(key=lambda x: x['center'][1])
    selected_rectangles = top_rectangles[:23]
    selected_rectangles.sort(key=lambda x: x['center'][0])

    return selected_rectangles


def find_side_black_rectangles(img):
    """좌우측 검은색 사각형들을 찾아서 좌표 반환 (Y축 계산용)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, upperValueSquare, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_height, img_width = img.shape[:2]
    left_area_threshold = img_width * 0.15
    right_area_start = img_width * 0.85

    left_rectangles = []
    right_rectangles = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + w // 2
        center_y = y + h // 2

        if (w > 20 and h > 20 and
            w < 70 and h < 70):

            if center_x < left_area_threshold:
                left_rectangles.append({
                    'center': (center_x, center_y),
                    'bbox': (x, y, w, h),
                    'area': w * h
                })
            elif center_x > right_area_start:
                right_rectangles.append({
                    'center': (center_x, center_y),
                    'bbox': (x, y, w, h),
                    'area': w * h
                })

    left_rectangles.sort(key=lambda x: x['center'][0])
    left_selected = left_rectangles[:10]
    left_selected.sort(key=lambda x: x['center'][1])

    right_rectangles.sort(key=lambda x: x['center'][0])
    right_selected = right_rectangles[-20:] if len(right_rectangles) >= 20 else right_rectangles
    right_selected.sort(key=lambda x: x['center'][1])

    return left_selected, right_selected


def define_phone_positions(img_width, img_height, top_rectangles, left_rectangles):
    """전화번호 전용 위치 계산 (1-8번 사각형, 1-10번 숫자 자리)"""
    phone_positions = {}

    for rect_index in range(8):
        if rect_index < len(top_rectangles):
            rect_center_x = top_rectangles[rect_index]['center'][0]
            digit_position = rect_index + 1

            if len(left_rectangles) >= 10:
                phone_positions[digit_position] = {}
                for digit in range(10):
                    if digit < len(left_rectangles):
                        y = left_rectangles[digit]['center'][1]
                        phone_positions[digit_position][str(digit)] = (rect_center_x, y)

    return phone_positions


def define_answer_positions(img_width, img_height, top_rectangles, left_rectangles, right_rectangles):
    """답안 전용 위치 계산 (9-23번 사각형, 1-45번 문제)"""
    positions = {}

    for rect_index in range(8, 13):
        if rect_index < len(top_rectangles):
            rect_center_x = top_rectangles[rect_index]['center'][0]
            choice_num = rect_index - 7

            if len(right_rectangles) >= 20:
                for q in range(1, 21):
                    if q not in positions:
                        positions[q] = {}

                    question_index = q - 1
                    if question_index < len(right_rectangles):
                        y = right_rectangles[question_index]['center'][1]
                        positions[q][str(choice_num)] = (rect_center_x, y)

    for rect_index in range(13, 18):
        if rect_index < len(top_rectangles):
            rect_center_x = top_rectangles[rect_index]['center'][0]
            choice_num = rect_index - 12

            if len(right_rectangles) >= 20:
                for q in range(21, 41):
                    if q not in positions:
                        positions[q] = {}

                    question_index = q - 21
                    if question_index < len(right_rectangles):
                        y = right_rectangles[question_index]['center'][1]
                        positions[q][str(choice_num)] = (rect_center_x, y)

    for rect_index in range(18, 23):
        if rect_index < len(top_rectangles):
            rect_center_x = top_rectangles[rect_index]['center'][0]
            choice_num = rect_index - 17

            if len(right_rectangles) >= 5:
                for q in range(41, 46):
                    if q not in positions:
                        positions[q] = {}

                    question_index = q - 41
                    if question_index < len(right_rectangles):
                        y = right_rectangles[question_index]['center'][1]
                        positions[q][str(choice_num)] = (rect_center_x, y)

    return positions


def estimate_phone_number_with_density(img, phone_positions, min_density=0.17):
    """전화번호 추정"""
    phone_selected = {}

    for digit_pos, digit_choices in phone_positions.items():
        if not digit_choices:
            phone_selected[digit_pos] = "0"
            continue

        digit_densities = {}
        for digit, coord in digit_choices.items():
            x, y = coord
            density = calculate_marking_density(img, x, y)
            digit_densities[digit] = density

        highest_digit, highest_density = max(digit_densities.items(), key=lambda x: x[1])

        if highest_density >= min_density:
            phone_selected[digit_pos] = highest_digit
        else:
            phone_selected[digit_pos] = "0"

    return phone_selected


def estimate_selected_answers_with_density(img, answer_positions, min_density=0.2):
    """답안 추정"""
    selected = {}

    for q_num, choices in answer_positions.items():
        if not choices:
            selected[str(q_num)] = "무효"
            continue

        choice_densities = {}
        for choice, coord in choices.items():
            x, y = coord
            density = calculate_marking_density(img, x, y)
            choice_densities[choice] = density

        highest_choice, highest_density = max(choice_densities.items(), key=lambda x: x[1])

        if highest_density >= min_density:
            selected[str(q_num)] = highest_choice
        else:
            selected[str(q_num)] = "무효"

    return selected


def extract_phone_number(phone_selected):
    """전화번호 8자리 추출"""
    phone_digits = []

    for i in range(1, 9):
        if i in phone_selected:
            digit = phone_selected[i]
            if digit and digit != "무효":
                try:
                    digit_int = int(digit)
                    if 0 <= digit_int <= 9:
                        phone_digits.append(str(digit_int))
                    else:
                        phone_digits.append("0")
                except ValueError:
                    phone_digits.append("0")
            else:
                phone_digits.append("0")
        else:
            phone_digits.append("0")

    phone_number = "".join(phone_digits)
    return phone_number


def calculate_total_score(selected_answers, correct_answers, question_scores):
    """총점 계산"""
    total = 0
    
    for q_num, correct_answer in correct_answers.items():
        if q_num in selected_answers:
            student_answer = selected_answers[q_num]
            if student_answer == correct_answer:
                score = question_scores.get(q_num, 0)
                total += score
    
    return total


def calculate_grade(total_score):
    """등급 계산"""
    if total_score >= 90:
        return 1
    elif total_score >= 80:
        return 2
    elif total_score >= 70:
        return 3
    elif total_score >= 60:
        return 4
    elif total_score >= 50:
        return 5
    elif total_score >= 40:
        return 6
    elif total_score >= 30:
        return 7
    elif total_score >= 20:
        return 8
    else:
        return 9


def create_results_array(selected_answers, correct_answers, question_scores, question_types):
    """결과 배열 생성"""
    results = []
    
    for q_num in sorted(correct_answers.keys(), key=lambda x: int(x)):
        try:
            q_num_int = int(q_num)
            student_answer = selected_answers.get(q_num, "무효")
            correct_answer = correct_answers[q_num]
            score = question_scores.get(q_num, 0)
            question_type = question_types.get(q_num, "기타")
            
            earned_score = score if student_answer == correct_answer else 0
            
            results.append({
                "questionNumber": q_num_int,
                "studentAnswer": str(student_answer),
                "correctAnswer": correct_answer,
                "score": score,
                "earnedScore": earned_score,
                "questionType": question_type
            })
        except (ValueError, TypeError):
            continue
    
    return results


def grade_omr_from_bytes(image_bytes: bytes, correct_answers: Dict, question_scores: Dict, question_types: Dict):
    """OMR 채점 메인 함수 (바이트 입력)"""
    try:
        img, (h, w) = load_image_from_bytes(image_bytes)
        deskewed_img = deskew_image_with_barcodes(img)

        expected_ratio = TARGET_WIDTH / TARGET_HEIGHT
        actual_ratio = w / h

        resized_img, scale_x, scale_y = resize_image_to_target(deskewed_img)
        resized_h, resized_w = resized_img.shape[:2]

        preprocessed_img = preprocess_omr_image(resized_img)

        top_rectangles = find_top_black_rectangles(resized_img)
        left_rectangles, right_rectangles = find_side_black_rectangles(resized_img)

        phone_positions = define_phone_positions(resized_w, resized_h, top_rectangles, left_rectangles)
        answer_positions = define_answer_positions(resized_w, resized_h, top_rectangles, left_rectangles, right_rectangles)

        phone_selected = estimate_phone_number_with_density(preprocessed_img, phone_positions)
        phone_number = extract_phone_number(phone_selected)

        selected_answers = estimate_selected_answers_with_density(preprocessed_img, answer_positions)

        correct_count = 0
        for q_num, correct_answer in correct_answers.items():
            if q_num in selected_answers:
                student_answer = selected_answers[q_num]
                if str(student_answer) == str(correct_answer):
                    correct_count += 1

        total_score = calculate_total_score(selected_answers, correct_answers, question_scores)
        grade = calculate_grade(total_score)
        results = create_results_array(selected_answers, correct_answers, question_scores, question_types)

        final_result = {
            "totalScore": total_score,
            "grade": grade,
            "phoneNumber": phone_number,
            "results": results,
            "imageInfo": {
                "originalSize": f"{w}x{h}",
                "resizedSize": f"{resized_w}x{resized_h}",
                "scaleFactors": {"x": scale_x, "y": scale_y},
                "aspectRatio": {"expected": expected_ratio, "actual": actual_ratio}
            }
        }

        return final_result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OMR 채점 실패: {str(e)}")


# API 엔드포인트들

@app.get("/")
async def root():
    """헬스체크 엔드포인트"""
    return {"status": "healthy", "message": "Joossam OMR Grading API is running"}


@app.get("/health")
async def health():
    """헬스체크 엔드포인트"""
    return {"status": "healthy"}


class GradingRequest(BaseModel):
    correct_answers: Dict[str, str]
    question_scores: Dict[str, int]
    question_types: Dict[str, str]


@app.post("/api/omr/grade")
async def grade_omr(
    image: UploadFile = File(...),
    correct_answers: str = Form(...),
    question_scores: str = Form(...),
    question_types: str = Form(...)
):
    """
    OMR 채점 API
    
    - image: OMR 이미지 파일
    - correct_answers: 정답 JSON (예: {"1": "3", "2": "1", ...})
    - question_scores: 문제별 점수 JSON (예: {"1": 2, "2": 2, ...})
    - question_types: 문제 유형 JSON (예: {"1": "어휘", "2": "문법", ...})
    """
    try:
        # JSON 파싱
        correct_answers_dict = json.loads(correct_answers)
        question_scores_dict = json.loads(question_scores)
        question_types_dict = json.loads(question_types)
        
        # 이미지 읽기
        image_bytes = await image.read()
        
        # OMR 채점 실행
        result = grade_omr_from_bytes(
            image_bytes,
            correct_answers_dict,
            question_scores_dict,
            question_types_dict
        )
        
        return result
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"JSON 파싱 오류: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"채점 오류: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

