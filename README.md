# ML
# 가나다 (AI-Powered Pronunciation Learning Solution)

<p align="center">
  <img src="path_to_project_logo.png" alt="가나다 로고" width="200">
</p>

> 💡 COVID-19로 인한 마스크 착용 의무화로 아이들의 발음 학습에 어려움을 해결하기 위한 인공지능 기반 비대면 교육 솔루션

## 🏆 수상
- 제 5회 KB국민은행 소프트웨어 경진대회 특별상

## 📋 프로젝트 소개
발음 학습의 어려움을 겪는 아이들을 위해 LipNet 기술을 활용한 비대면 교육 솔루션입니다. 마스크 착용으로 인한 발음 교육의 한계를 극복하고, 효과적인 학습을 도와주는 애플리케이션을 개발했습니다.

### 주요 기능
- **머신러닝**
  - 자음/모음 학습 데이터 생성
  - 발음 정확도 측정
  - 자세 교정 안내 (올바른 발음을 위한 자세 유도)
- **안드로이드 앱**
  - 회원가입/로그인
  - 학습하기
  - 연습하기
  - 학습 아동 세부 관리
  - 커뮤니티 (공지사항, 게시판)

## 🛠 기술 스택
- **AI/ML**: LipNet, TensorFlow
- **Mobile**: Android (Java/Kotlin)
- **Backend**: [사용된 백엔드 기술]
- **DevOps**: [배포 및 운영 기술]

## 🔍 모델 개발 프로세스
1. **데이터 준비**
   - 학습용 자음/모음 동영상 촬영 (3초 ± 0.2초)
   - 75 Frame으로 분리하여 데이터셋 구성
   
2. **모델 학습**
   - LipNet 아키텍처 기반 학습 진행
   - 각 프레임별 발음 레이블링
   - 학습 결과 가중치 모델 생성 (weight.h5)

3. **성능 평가**
   - Train, Validation, Test 데이터셋 기반 평가
   - 정확도 측정 및 개선점 도출

## 👥 팀 구성
- **팀명**: SSU Cream
- **인원**: 총 4명
  - PM & Deep Learning (기획 80%, 모델 개발 50%)
  - Frontend Developer
  - Backend Developer
  - Deep Learning Engineer

## ⚙️ 설치 및 실행방법
```bash
# 저장소 클론
git clone https://github.com/SSU-Cream/ML

# 의존성 설치
[설치 명령어]

# 실행 방법
[실행 명령어]
```

## 📝 라이센스
이 프로젝트는 LipNet을 기반으로 개발되었습니다.
- Original LipNet: https://github.com/rizkiarm/LipNet

## 🔄 향후 개선사항
- 모델 학습을 위한 추가 데이터셋 확보 필요
- 안드로이드 UX/UI 최적화
- 교사용 기능 개선 및 확장

## 📅 프로젝트 기간
2022.02 - 2022.11

## 🔗 관련 링크
- [프로젝트 저장소](https://github.com/SSU-Cream)
- [발표 자료]
- [시연 영상]
