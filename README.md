# auv_vla_reasoning

KMU26 **실물 AUV용 VLA 추론** 저장소입니다. 기존 `auv_vla`의 U0 추론 코드에서
모델 실행, 센서 입력, 조종값 전달을 분리했습니다. PPO 보정과 시뮬레이터 설정은 포함하지 않습니다.

```text
실물 카메라·DVL·IMU·수심 + 작업 설명
    → ROS 관측 노드 → HTTP 추론 서버 → 16 × 4 조종값
    → 시간·모드·조종 범위 검사 → 확인용 토픽 또는 MAVROS
```

시작 상태는 **비활성 + dry_run**입니다. 모델 가중치는 별도로 준비해야 합니다.
서버는 기체를 ARM하거나 모드를 바꾸지 않으며, ROS 어댑터도 이를 요청하지 않습니다.

## 기존 네 저장소와의 역할

| 저장소 | 담당 기능 | 이 저장소와 연결되는 부분 |
|---|---|---|
| [auv_vla_training](https://github.com/2026-kmu-underwater-robot/auv_vla_training) | `kmu_auv` 모델 학습 | 19개 상태와 학습 시 min-max 전처리 |
| [auv_vla](https://github.com/2026-kmu-underwater-robot/auv_vla) | U0 모델·학습 | 기존 23개 상태 체크포인트와 전처리 |
| [auv_vla_gui](https://github.com/2026-kmu-underwater-robot/auv_vla_gui) | ROV·카메라 실행, 상태 표시 | GUI가 실행한 장비의 ROS 토픽 구독 |
| [auv_vla_data_collector](https://github.com/2026-kmu-underwater-robot/auv_vla_data_collector) | 실물 시연 수집·U0 학습용 내보내기 | 23개 상태 저장, `kmu_auv`에서 19개 선택, 센서 변환 기준 |
| [auv_vla_dataset_maker](https://github.com/2026-kmu-underwater-robot/auv_vla_dataset_maker) | 별도 수집 대시보드·LeRobot v3 내보내기 | 현재 U0 학습 로더는 v3를 직접 읽지 않음 |

현재 GUI는 ROV/RealSense 런치만 실행합니다. 추론 버튼은 아직 없으므로 아래 명령으로
이 저장소를 별도로 실행합니다. GUI의 ROV 런치를 다시 띄우거나 종료하지 않습니다.

데이터셋 제작기와 기존 수집기의 ROS 패키지 이름은 둘 다
`kmu26_auv_vla_data_collector`입니다. 이 저장소는 그 패키지를 설치하거나 실행하지 않고
`vla_observation`이라는 별도 노드에서 센서만 읽습니다. 녹화 서비스와 파일 저장도 없습니다.
수집 기준은 `auv_vla_data_collector`의 `159fa7935aafb413390b9d49782e5eef093a10a8`입니다.
해당 버전은 에피소드별 v2.0 구조를 출력하고 학습 코드의 `kmu_auv` 필드 매핑을 제공합니다.
실제 전체 모델 학습의 성공까지 확인한 것은 아닙니다.

## 코드 위치

| 경로 | 내용 |
|---|---|
| `auv_vla_reasoning/server.py` | 모델 로딩, HTTP `/act`, `/metadata`, `/health` |
| `auv_vla_reasoning/checkpoint.py` | 실물 모델 설정·전처리·출력 형식 사전 검사 |
| `auv_vla_reasoning/observations.py` | 실물 센서 구독과 모델 입력 생성 |
| `auv_vla_reasoning/sensor_math.py` | 좌표축 변환, 자세 정규화, 23개 상태값 구성 |
| `auv_vla_reasoning/ros.py` | 활성화, 입력 유효성, 조종값 제한, MAVROS 연결 |
| `auv_vla_reasoning/controls.py` | 4축 출력 검사와 PWM 변환 |
| `auv_vla_reasoning/client.py` | 저장된 NPZ 입력으로 HTTP 추론 확인 |
| `auv_vla_reasoning/_vendor/gr00t/` | 기존 U0 추론에 필요한 공통 모델·전처리 코드 |
| `config/real.yaml` | 실제 장비의 ROS 토픽·조종 설정 |

모델 코드가 공유하는 `ModalityConfig` 등의 자료형 때문에 일부 공통 데이터 모듈은
포함되어 있습니다. 학습 실행 파일, 데이터 변환기, 녹화기, GUI, 모델 가중치는 포함하지 않습니다.
원본과 반영한 수정은 [소스 검토 기록](docs/SOURCE_REVIEW.md)을 참고하세요.

## 1. GPU 추론 환경

ROS Humble 환경과 GPU 모델 환경을 분리합니다. 기존 `auv_vla` 모델이 실행되는
Python 3.10 이상 환경에서 이 저장소를 설치하는 것이 가장 간단합니다.

```bash
git clone https://github.com/2026-kmu-underwater-robot/auv_vla_reasoning.git
cd auv_vla_reasoning
python -m pip install -e '.[server]'
```

새 GPU 환경은 하드웨어에 맞는 PyTorch·torchvision·FlashAttention·PyTorch3D를 먼저
준비한 뒤 `python -m pip install -r requirements-model.txt`를 실행합니다.
기존 모델의 환경 정의는 [원본 설정 사본](docs/upstream-model-pyproject.toml)에 있습니다.
이 저장소의 기본 설치는 대용량 CUDA 패키지나 모델 가중치를 자동으로 받지 않습니다.

### `kmu_auv` 방식으로 학습한 실물 모델

기본 선택은 `auv_vla_training`의 `kmu_auv`입니다. 모델 입력은 자세(4), DVL 속도(3),
수심(1), 이전 명령(4), 각속도(3), 가속도(3), 고도(1) 순서의 19개 물리값입니다.
학습 설정과 같은 영상 처리와 각 상태·행동의 min-max 정규화를 사용하며, 통계는
해당 체크포인트의 `experiment_cfg/metadata.json`에서 읽습니다. 상태 4개의 유효성은
ROS/HTTP에서 검사하되 이 모델에 전달하지 않습니다. 이 모델은 DVL 무효를 구분할 수
없으므로 두 카메라, DVL 속도 및 고도가 모두 유효할 때만 추론합니다.

체크포인트에는 조종 설정과 학습 전처리가 자동으로 증명되지 않으므로
`config/runtime.example.json`을 복사해 **학습 시 사용한 실제 설정**에 맞게 작성합니다.
학습 기록에서 `kmu_auv` 사용을 확인한 모델에만 이 설정을 선택하세요. 예제나 메타데이터
형식 검사는 학습 출처를 증명하지 않습니다.

```bash
python -m auv_vla_reasoning.server \
  --checkpoint /path/to/real-checkpoint \
  --data_config kmu_auv \
  --runtime_config /path/to/reviewed-runtime.json \
  --check_only

python -m auv_vla_reasoning.server \
  --checkpoint /path/to/real-checkpoint \
  --data_config kmu_auv \
  --runtime_config /path/to/reviewed-runtime.json \
  --host 127.0.0.1 --port 8000 --device cuda
```

체크포인트에는 모델 가중치, `config.json`, `experiment_cfg/metadata.json`이 필요합니다.
원본 U0의 다른 로봇용 가중치나 시뮬레이터 전용 체크포인트는 이 경로에서 거부합니다.
부표 분리 위치 예측(CAP)은 꺼진 모델만 지원합니다.
특히 학습 코드가 CAP 비활성을 실행 중 설정에만 적용하고 `config.json`에 저장하지 않은
체크포인트는 서버가 거부할 수 있습니다. 실제 학습 설정과 저장/재로딩 결과를 확인해야 하며,
파일의 값을 임의로 바꾸는 것만으로 CAP 비활성 학습이 증명되지는 않습니다.

### 기존 23개 상태 모델

`auv_vla`의 `kmu26_auv_real`로 학습한 모델은 `--data_config kmu26_auv_real`을 명시하고,
별도 runtime JSON의 `data_config`도 동일하게 바꿔 사용합니다. 기존 46개 sin/cos 입력을
유지합니다. `kmu_auv`와 서로 다른 전처리이므로 체크포인트에 맞춰 선택해야 합니다.

### 수정된 v2 방식으로 새로 학습한 모델

기존 모델 PR의 `kmu26_auv_real_v2`로 학습했다면 `--data_config kmu26_auv_real_v2`를
명시합니다. `config.json`의 `kmu26_training` 설정을 읽으므로 별도 runtime JSON은
필수가 아닙니다. 두 방식의 전처리는 다르며, 기존 가중치를 실행하면서 이름만 v2로
바꾸면 안 됩니다. 기본 선택은 `kmu_auv`입니다.

GPU 서버와 로봇 PC가 다르면 접근 가능한 서버 주소에 바인딩하고
`config/real.yaml`의 `server_url`을 그 주소로 바꿉니다. HTTP API에는 인증 기능이
없으므로 운용 네트워크 내부에서 사용합니다.

## 2. 로봇 PC의 ROS 환경

필요한 ROS 메시지는 Humble의 `sensor_msgs`, `geometry_msgs`, `mavros_msgs`,
`std_msgs`, `std_srvs`와 장비 워크스페이스의 `auv_dvl_a50_msg`입니다.
이 환경에는 torch가 필요하지 않습니다.

```bash
source /opt/ros/humble/setup.bash
source /path/to/auv_ros2/install/setup.bash
python3 -m venv --system-site-packages .venv-ros
.venv-ros/bin/python -m pip install -e .
.venv-ros/bin/python -m auv_vla_reasoning.ros \
  --ros-args --params-file config/real.yaml
```

ROS 워크스페이스의 `src/auv_vla_reasoning`에 클론한 경우 colcon 실행도 지원합니다.
colcon이 사용하는 Python에 `json-numpy`와 `requests`가 필요합니다.

```bash
colcon build --packages-select auv_vla_reasoning --symlink-install
source install/setup.bash
ros2 launch auv_vla_reasoning reasoning.launch.py
```

센서와 추론 노드 모두 `use_sim_time: false`를 사용합니다. 실물 노드는 시뮬레이션 시간
설정을 거부합니다. 토픽이나 센서 설정을 바꾸면 노드를 재시작합니다.

### 카메라와 상태값

기본 카메라는 기존 학습 데이터와 같은 IMX219 compressed RGB 두 뷰입니다.
GUI의 RealSense 런치가 실행됐다고 전방·분리부 카메라가 모두 준비된 것은 아닙니다.
학습과 같은 시야의 raw RGB 카메라를 쓰는 경우 해당 뷰의 `*_image_topic`을 바꾸고
`*_image_transport: raw`를 지정합니다. `rgb8`, `bgr8`을 지원하며 프레임을 RGB로 전달합니다.

| 입력 | ROS 관측 노드가 수집하는 값 |
|---|---|
| 이전 조종값 | `[surge, sway, heave, yaw]` 4개 |
| DVL | FLU 속도 3개, 고도 1개, 각각 유효성 표시 |
| IMU | 각속도 3개, 선형가속도 3개, 정규화된 wxyz 자세 4개 |
| 수심 | 아래 방향이 양수인 m 단위 값 1개 |
| 유효성 | 카메라 두 개, DVL 속도, 고도 총 4개 |

카메라 두 개·IMU·수심은 최신 값이 있어야 합니다. DVL이 없거나 무효하면 관련 값은
0, 유효성은 0으로 표시합니다. `kmu_auv`에서는 이 상태를 모델에 보내지 않고 추론을
거부합니다. `dvl_link`의 FRD → FLU 변환은 기존 수집기와 같습니다.
다른 센서 장착 방향은 먼저 올바른 body frame으로 변환해야 합니다.

### 추론 활성화와 출력

작업 설명은 `/vla/task_description` (`std_msgs/msg/String`)에 전달합니다.
사용자가 누르고 있는 제어장치가 `/vla/deadman` (`std_msgs/msg/Bool`)에 true를
10 Hz로 보내는 동안 `/vla/enable` (`std_srvs/srv/SetBool`)을 true로 호출하면 활성화됩니다.
작업 설명 변경, 센서 지연, 서버 오류, deadman 중단 시 비활성화되며 다시 활성화해야 합니다.

기본 출력은 `/vla/proposed_rc`입니다. 실제 RC 출력은 `dry_run: false`일 때만
`/mavros/rc/override`로 나갑니다. 연결·ARM·STABILIZE 상태와 서버 설정 일치를 확인하며,
같은 RC 토픽에 다른 발행자가 보이면 실행을 거부/중단합니다. 기존 GUI가 시작한
`joy2mavros`와 제어권을 동시에 사용하지 않도록 운용 측에서 조정해야 합니다.
ROS 발행자 확인은 완전한 RC 중재 장치를 대신하지 않습니다.

모델은 10 Hz 기준 16개 명령을 예측하지만, 어댑터는 기본 0.3초 이내의 명령만 사용합니다.
추론에 걸린 시간에 해당하는 앞쪽 명령은 건너뜁니다. 실제 서버의 응답 시간이 이 범위에
들어오는지 확인해야 합니다. 정지할 때는 소유한 `[5, 6, 3, 4]` 채널만 해제합니다.
부표 분리 액추에이터 제어는 포함하지 않습니다.

## 검증

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

일반 Python에서는 모델 설정·HTTP·출력 검사 테스트가 실행됩니다. ROS 테스트는
Humble과 해당 메시지 패키지가 준비된 환경에서 실행하며, 실제 센서·기체를 사용하지 않습니다.
테스트 범위와 실제 장비에서 남은 확인은 [검증 기록](docs/VALIDATION.md)을 참고하세요.
