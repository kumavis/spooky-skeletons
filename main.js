import * as THREE from "three";
import {
  FilesetResolver,
  PoseLandmarker,
} from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.7";
import { TDSLoader } from "three/addons/loaders/TDSLoader.js";

const MODEL_ASSET =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task";
const WASM_ASSET =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm";

const canvas = document.getElementById("scene");
const cameraButton = document.getElementById("cameraButton");
const statusLabel = document.getElementById("status");
const videoElement = document.getElementById("hiddenVideo");

const renderModeSelect = document.getElementById("renderMode");
const smoothingSlider = document.getElementById("smoothing");
const scaleXSlider = document.getElementById("scaleX");
const scaleYSlider = document.getElementById("scaleY");
const scaleZSlider = document.getElementById("scaleZ");
const mirrorCheckbox = document.getElementById("mirror");
const showRigCheckbox = document.getElementById("showRig");
const showDebugSpheresCheckbox = document.getElementById("showDebugSpheres");
const boneRotationStrategySelect = document.getElementById("boneRotationStrategy");
const uiElement = document.getElementById("ui");
const posXSlider = document.getElementById("posX");
const posYSlider = document.getElementById("posY");
const posZSlider = document.getElementById("posZ");
const rotXSlider = document.getElementById("rotX");
const rotYSlider = document.getElementById("rotY");
const rotZSlider = document.getElementById("rotZ");
const restPoseCheckbox = document.getElementById("restPose");
const debugButton = document.getElementById("debugButton");
const debugPanel = document.getElementById("debugPanel");
const debugStepLabel = document.getElementById("debugStepLabel");
const debugSegmentName = document.getElementById("debugSegmentName");
const debugInfo = document.getElementById("debugInfo");
const debugStartSelect = document.getElementById("debugStartSelect");
const debugEndSelect = document.getElementById("debugEndSelect");
const debugAxisSelect = document.getElementById("debugAxisSelect");
const debugNotes = document.getElementById("debugNotes");
const debugPrev = document.getElementById("debugPrev");
const debugNext = document.getElementById("debugNext");
const debugClose = document.getElementById("debugClose");
const debugDump = document.getElementById("debugDump");
const debugPreviewApply = document.getElementById("debugPreviewApply");

const scaleXValue = document.getElementById("scaleXValue");
const scaleYValue = document.getElementById("scaleYValue");
const scaleZValue = document.getElementById("scaleZValue");
const smoothingValue = document.getElementById("smoothingValue");
const posXValue = document.getElementById("posXValue");
const posYValue = document.getElementById("posYValue");
const posZValue = document.getElementById("posZValue");
const rotXValue = document.getElementById("rotXValue");
const rotYValue = document.getElementById("rotYValue");
const rotZValue = document.getElementById("rotZValue");

const calibration = {
  smoothing: parseFloat(smoothingSlider.value),
  scaleX: parseFloat(scaleXSlider.value),
  scaleY: parseFloat(scaleYSlider.value),
  scaleZ: parseFloat(scaleZSlider.value),
  mirror: mirrorCheckbox.checked,
  positionX: parseFloat(posXSlider.value),
  positionY: parseFloat(posYSlider.value),
  positionZ: parseFloat(posZSlider.value),
  rotationX: parseFloat(rotXSlider.value),
  rotationY: parseFloat(rotYSlider.value),
  rotationZ: parseFloat(rotZSlider.value),
};

const calibrationDefaults = {
  positionX: calibration.positionX,
  positionY: calibration.positionY,
  positionZ: calibration.positionZ,
  rotationX: calibration.rotationX,
  rotationY: calibration.rotationY,
  rotationZ: calibration.rotationZ,
};

const JOINT_COUNT = 33;
const bones = [];
const joints = [];
const jointPositions = Array.from(
  { length: JOINT_COUNT },
  () => new THREE.Vector3(),
);

const BONE_PAIRS = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 7],
  [0, 4],
  [4, 5],
  [5, 6],
  [6, 8],
  [9, 10],
  [11, 12],
  [11, 13],
  [13, 15],
  [15, 17],
  [15, 19],
  [15, 21],
  [12, 14],
  [14, 16],
  [16, 18],
  [16, 20],
  [16, 22],
  [11, 23],
  [12, 24],
  [23, 24],
  [23, 25],
  [25, 27],
  [27, 29],
  [29, 31],
  [24, 26],
  [26, 28],
  [28, 30],
  [30, 32],
];

// Bone rotation configuration for pole vector and anatomical strategies
const BONE_ROTATION_CONFIG = {
  // Left arm: shoulder(11) -> elbow(13) uses wrist(15) as pole
  "11-13": { pole: 15, anatomical: new THREE.Vector3(0, 0, 1) }, // Forward
  // Left arm: elbow(13) -> wrist(15) uses index(19) as pole
  "13-15": { pole: 19, anatomical: new THREE.Vector3(0, 0, 1) }, // Forward
  // Right arm: shoulder(12) -> elbow(14) uses wrist(16) as pole
  "12-14": { pole: 16, anatomical: new THREE.Vector3(0, 0, 1) }, // Forward
  // Right arm: elbow(14) -> wrist(16) uses index(20) as pole
  "14-16": { pole: 20, anatomical: new THREE.Vector3(0, 0, 1) }, // Forward
  // Left leg: hip(23) -> knee(25) uses ankle(27) as pole
  "23-25": { pole: 27, anatomical: new THREE.Vector3(0, 0, -1) }, // Backward
  // Left leg: knee(25) -> ankle(27) uses heel(29) as pole
  "25-27": { pole: 29, anatomical: new THREE.Vector3(0, 0, -1) }, // Backward
  // Right leg: hip(24) -> knee(26) uses ankle(28) as pole
  "24-26": { pole: 28, anatomical: new THREE.Vector3(0, 0, -1) }, // Backward
  // Right leg: knee(26) -> ankle(28) uses heel(30) as pole
  "26-28": { pole: 30, anatomical: new THREE.Vector3(0, 0, -1) }, // Backward
  // Torso: shoulder(11) -> hip(23)
  "11-23": { pole: 12, anatomical: new THREE.Vector3(0, 0, -1) }, // Backward
  "12-24": { pole: 11, anatomical: new THREE.Vector3(0, 0, -1) }, // Backward
};

let renderer;
let scene;
let camera;
let poseLandmarker;
let lastVideoTime = -1;
let animationFrameId;
let renderMode = renderModeSelect.value;
let boneRotationStrategy = boneRotationStrategySelect.value;
let rigContainer;
let jointGroup;
let assetGroup = null;
let assetGroupAlt = null;
let assetOriginalHeight = 1;
let assetOriginalHeightAlt = 1;
let assetLoaded = false;
let assetLoadedAlt = false;
let assetLastScale = parseFloat(scaleYSlider.value);
let assetFootOffset = 0;
const assetRestTransform = {
  position: new THREE.Vector3(),
  quaternion: new THREE.Quaternion(),
  scale: new THREE.Vector3(1, 1, 1),
};
const assetRestTransformAlt = {
  position: new THREE.Vector3(),
  quaternion: new THREE.Quaternion(),
  scale: new THREE.Vector3(1, 1, 1),
};
let assetRestReady = false;
let assetRestReadyAlt = false;
let savedCalibrationForRest = null;
let overlayJointRig = Boolean(
  showRigCheckbox ? showRigCheckbox.checked : false,
);
let showDebugSpheres = Boolean(
  showDebugSpheresCheckbox ? showDebugSpheresCheckbox.checked : false,
);

const jointVisibility = new Array(JOINT_COUNT).fill(false);

const yAxis = new THREE.Vector3(0, 1, 0);
const xAxis = new THREE.Vector3(1, 0, 0);
const zAxis = new THREE.Vector3(0, 0, 1);
const tempStart = new THREE.Vector3();
const tempEnd = new THREE.Vector3();
const tempDir = new THREE.Vector3();
const tempHipCenter = new THREE.Vector3();
const tempShoulderCenter = new THREE.Vector3();
const tempLeftRight = new THREE.Vector3();
const tempForward = new THREE.Vector3();
const tempUp = new THREE.Vector3();
const tempFeet = new THREE.Vector3();
const tempHead = new THREE.Vector3();
const assetPivotOffset = new THREE.Vector3();
const assetMatrix = new THREE.Matrix4();
const assetBaseQuaternion = new THREE.Quaternion();
const tempDirLocal = new THREE.Vector3();
const tempAlignQuat = new THREE.Quaternion();
const tempScaleVector = new THREE.Vector3();
const tempPoleVec = new THREE.Vector3();
const tempRight = new THREE.Vector3();
const tempUpVec = new THREE.Vector3();
const tempMatrix = new THREE.Matrix4();
const debugLinePositions = new Float32Array(6);
let debugLine = null;

const LANDMARK_LABELS = [
  "Nose",
  "Left eye inner",
  "Left eye",
  "Left eye outer",
  "Right eye inner",
  "Right eye",
  "Right eye outer",
  "Left ear",
  "Right ear",
  "Mouth left",
  "Mouth right",
  "Left shoulder",
  "Right shoulder",
  "Left elbow",
  "Right elbow",
  "Left wrist",
  "Right wrist",
  "Left pinky",
  "Right pinky",
  "Left index",
  "Right index",
  "Left thumb",
  "Right thumb",
  "Left hip",
  "Right hip",
  "Left knee",
  "Right knee",
  "Left ankle",
  "Right ankle",
  "Left heel",
  "Right heel",
  "Left foot index",
  "Right foot index",
];

const landmarkOptions = LANDMARK_LABELS.map((label, index) => ({
  label,
  value: index,
}));

const LM = {
  NOSE: 0,
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13,
  RIGHT_ELBOW: 14,
  LEFT_WRIST: 15,
  RIGHT_WRIST: 16,
  LEFT_INDEX: 19,
  RIGHT_INDEX: 20,
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
  LEFT_KNEE: 25,
  RIGHT_KNEE: 26,
  LEFT_ANKLE: 27,
  RIGHT_ANKLE: 28,
  LEFT_HEEL: 29,
  RIGHT_HEEL: 30,
  LEFT_FOOT_INDEX: 31,
  RIGHT_FOOT_INDEX: 32,
};

// Coordinate system transform: skeleton is Z-up, pose is Y-up
// This rotates from Z-up to Y-up (90° around X-axis)
const coordSystemTransform = new THREE.Quaternion().setFromAxisAngle(
  new THREE.Vector3(1, 0, 0),
  Math.PI / 2
);

const assetSegments = [
  // Arms
  {
    label: "Left lower arm (torso side)",
    name: "object_11",
    start: LM.LEFT_SHOULDER,
    end: LM.LEFT_ELBOW,
  },
  {
    label: "Left upper arm",
    name: "object_12",
    start: LM.LEFT_ELBOW,
    end: LM.LEFT_WRIST,
  },
  {
    label: "Left hand",
    name: "object_13",
    start: LM.LEFT_WRIST,
    end: LM.LEFT_INDEX,
  },
  {
    label: "Right lower arm (torso side)",
    name: "object_18",
    start: LM.RIGHT_SHOULDER,
    end: LM.RIGHT_ELBOW,
  },
  {
    label: "Right upper arm",
    name: "object_19",
    start: LM.RIGHT_ELBOW,
    end: LM.RIGHT_WRIST,
  },
  {
    label: "Right hand",
    name: "object_20",
    start: LM.RIGHT_WRIST,
    end: LM.RIGHT_INDEX,
  },
  // Legs
  {
    label: "Left thigh",
    name: "object_4",
    start: LM.LEFT_HIP,
    end: LM.LEFT_KNEE,
  },
  {
    label: "Left shin",
    name: "object_5",
    start: LM.LEFT_KNEE,
    end: LM.LEFT_ANKLE,
  },
  {
    label: "Left foot",
    name: "object_6",
    start: LM.LEFT_ANKLE,
    end: LM.LEFT_FOOT_INDEX,
  },
  {
    label: "Right thigh",
    name: "object_14",
    start: LM.RIGHT_HIP,
    end: LM.RIGHT_KNEE,
  },
  {
    label: "Right shin",
    name: "object_15",
    start: LM.RIGHT_KNEE,
    end: LM.RIGHT_ANKLE,
  },
  {
    label: "Right foot",
    name: "object_16",
    start: LM.RIGHT_ANKLE,
    end: LM.RIGHT_FOOT_INDEX,
  },
  // Head
  {
    label: "Neck",
    name: "object_7",
    start: LM.LEFT_SHOULDER,
    end: LM.NOSE,
  },
  {
    label: "Skull",
    name: "object_8",
    start: LM.NOSE,
    end: LM.NOSE,
  },
  {
    label: "Jaw",
    name: "object_9",
    start: LM.NOSE,
    end: LM.NOSE,
  },
  // Torso
  {
    label: "Left shoulder",
    name: "object_10",
    start: LM.LEFT_SHOULDER,
    end: LM.LEFT_SHOULDER,
  },
  {
    label: "Right shoulder",
    name: "object_17",
    start: LM.RIGHT_SHOULDER,
    end: LM.RIGHT_SHOULDER,
  },
  {
    label: "Chest",
    name: "object_3",
    start: LM.LEFT_SHOULDER,
    end: LM.LEFT_HIP,
  },
  {
    label: "Lower spine",
    name: "object_2",
    start: LM.LEFT_SHOULDER,
    end: LM.LEFT_HIP,
  },
  {
    label: "Pelvis",
    name: "object_1",
    start: LM.LEFT_HIP,
    end: LM.RIGHT_HIP,
  },
].map((segment) => ({
  ...segment,
  defaultStart: segment.start,
  defaultEnd: segment.end,
  userStart: segment.start,
  userEnd: segment.end,
  userAxisSelection: "auto",
  notes: "",
  mesh: null,
  meshAlt: null,
  baseScale: new THREE.Vector3(1, 1, 1),
  baseScaleAlt: new THREE.Vector3(1, 1, 1),
  axis: new THREE.Vector3(0, 1, 0),
  axisAlt: new THREE.Vector3(0, 1, 0),
  restLength: 1,
  restLengthAlt: 1,
  initialQuaternion: new THREE.Quaternion(),
  initialQuaternionAlt: new THREE.Quaternion(),
  restPosition: new THREE.Vector3(),
  restPositionAlt: new THREE.Vector3(),
  defaultAxis: new THREE.Vector3(0, 1, 0),
  defaultAxisAlt: new THREE.Vector3(0, 1, 0),
  originalEmissiveArray: null,
  originalEmissiveArrayAlt: null,
}));

const animatedSegmentNames = new Set([
  // Arms
  "object_11",
  "object_12",
  "object_13",
  "object_18",
  "object_19",
  "object_20",
  // Legs
  "object_4",
  "object_5",
  "object_6",
  "object_14",
  "object_15",
  "object_16",
  // Head
  "object_7",
  "object_8",
  "object_9",
  // Torso
  "object_1",
  "object_2",
  "object_3",
  "object_10",
  "object_17",
]);

const segmentDebugSpheres = new Map();

let debugMode = false;
let debugIndex = 0;
let highlightedSegment = null;
let landmarkSelectPopulated = false;
let poseTrackingEnabled = true;

initUI();
initThree();
animate();

cameraButton.addEventListener("click", async () => {
  try {
    await activateCamera();
    cameraButton.disabled = true;
    cameraButton.textContent = "Camera Enabled";
  } catch (err) {
    console.error(err);
    setStatus("Unable to access camera. Check permissions.", true);
  }
});

window.addEventListener("resize", () => {
  const { innerWidth, innerHeight } = window;
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

window.addEventListener("keydown", (event) => {
  // Tilde key (~) or backtick (`) toggles the menu
  if (event.key === "`" || event.key === "~") {
    if (uiElement) {
      uiElement.style.display = uiElement.style.display === "none" ? "" : "none";
    }
    event.preventDefault();
  }
});

async function activateCamera() {
  setStatus("Requesting camera access…");
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: "user",
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
    audio: false,
  });
  videoElement.srcObject = stream;
  videoElement.muted = true;
  videoElement.playsInline = true;
  await videoElement.play();

  setStatus("Loading pose estimator…");
  await loadPoseLandmarker();
  setStatus("Tracking pose");
}

async function loadPoseLandmarker() {
  if (poseLandmarker) {
    return;
  }
  const vision = await FilesetResolver.forVisionTasks(WASM_ASSET);
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: MODEL_ASSET,
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
}

function initUI() {
  updateSliderLabels();
  renderModeSelect.addEventListener("change", () => {
    renderMode = renderModeSelect.value;
    updateModelVisibility();
  });
  smoothingSlider.addEventListener("input", () => {
    calibration.smoothing = parseFloat(smoothingSlider.value);
    updateSliderLabels();
  });
  scaleXSlider.addEventListener("input", () => {
    calibration.scaleX = parseFloat(scaleXSlider.value);
    updateSliderLabels();
  });
  scaleYSlider.addEventListener("input", () => {
    calibration.scaleY = parseFloat(scaleYSlider.value);
    updateSliderLabels();
  });
  scaleZSlider.addEventListener("input", () => {
    calibration.scaleZ = parseFloat(scaleZSlider.value);
    updateSliderLabels();
  });
  mirrorCheckbox.addEventListener("change", () => {
    calibration.mirror = mirrorCheckbox.checked;
  });
  if (showRigCheckbox) {
    showRigCheckbox.addEventListener("change", () => {
      overlayJointRig = showRigCheckbox.checked;
      applyJointRigVisibility();
    });
  }
  if (showDebugSpheresCheckbox) {
    showDebugSpheresCheckbox.addEventListener("change", () => {
      showDebugSpheres = showDebugSpheresCheckbox.checked;
      updateDebugSpheresVisibility();
    });
  }
  if (boneRotationStrategySelect) {
    boneRotationStrategySelect.addEventListener("change", () => {
      boneRotationStrategy = boneRotationStrategySelect.value;
      console.log("Bone rotation strategy changed to:", boneRotationStrategy);
    });
  }
  posXSlider.addEventListener("input", () => {
    calibration.positionX = parseFloat(posXSlider.value);
    updateSliderLabels();
  });
  posYSlider.addEventListener("input", () => {
    calibration.positionY = parseFloat(posYSlider.value);
    updateSliderLabels();
  });
  posZSlider.addEventListener("input", () => {
    calibration.positionZ = parseFloat(posZSlider.value);
    updateSliderLabels();
  });
  rotXSlider.addEventListener("input", () => {
    calibration.rotationX = parseFloat(rotXSlider.value);
    updateSliderLabels();
  });
  rotYSlider.addEventListener("input", () => {
    calibration.rotationY = parseFloat(rotYSlider.value);
    updateSliderLabels();
  });
  rotZSlider.addEventListener("input", () => {
    calibration.rotationZ = parseFloat(rotZSlider.value);
    updateSliderLabels();
  });

  if (restPoseCheckbox) {
    restPoseCheckbox.addEventListener("change", () => {
      const enableRest = restPoseCheckbox.checked;
      poseTrackingEnabled = !enableRest;
      if (enableRest) {
        saveCurrentCalibration();
        applyCalibrationDefaults();
        setStatus("Pose tracking paused (rest pose)");
        applyRigTransform();
        applyRestPose();
      } else {
        restoreCalibrationAfterRest();
        setStatus("Tracking pose");
      }
    });
  }

  initDebugControls();
}

function updateSliderLabels() {
  smoothingValue.textContent = calibration.smoothing.toFixed(2);
  scaleXValue.textContent = calibration.scaleX.toFixed(2);
  scaleYValue.textContent = calibration.scaleY.toFixed(2);
  scaleZValue.textContent = calibration.scaleZ.toFixed(2);
  posXValue.textContent = calibration.positionX.toFixed(2);
  posYValue.textContent = calibration.positionY.toFixed(2);
  posZValue.textContent = calibration.positionZ.toFixed(2);
  rotXValue.textContent = `${calibration.rotationX.toFixed(0)}°`;
  rotYValue.textContent = `${calibration.rotationY.toFixed(0)}°`;
  rotZValue.textContent = `${calibration.rotationZ.toFixed(0)}°`;
}

function updateSliderElementsFromCalibration() {
  posXSlider.value = calibration.positionX.toString();
  posYSlider.value = calibration.positionY.toString();
  posZSlider.value = calibration.positionZ.toString();
  rotXSlider.value = calibration.rotationX.toString();
  rotYSlider.value = calibration.rotationY.toString();
  rotZSlider.value = calibration.rotationZ.toString();
}

function saveCurrentCalibration() {
  savedCalibrationForRest = {
    positionX: calibration.positionX,
    positionY: calibration.positionY,
    positionZ: calibration.positionZ,
    rotationX: calibration.rotationX,
    rotationY: calibration.rotationY,
    rotationZ: calibration.rotationZ,
  };
}

function applyCalibrationDefaults() {
  calibration.positionX = calibrationDefaults.positionX;
  calibration.positionY = calibrationDefaults.positionY;
  calibration.positionZ = calibrationDefaults.positionZ;
  calibration.rotationX = calibrationDefaults.rotationX;
  calibration.rotationY = calibrationDefaults.rotationY;
  calibration.rotationZ = calibrationDefaults.rotationZ;
  updateSliderElementsFromCalibration();
  updateSliderLabels();
}

function restoreCalibrationAfterRest() {
  if (!savedCalibrationForRest) {
    return;
  }
  calibration.positionX = savedCalibrationForRest.positionX;
  calibration.positionY = savedCalibrationForRest.positionY;
  calibration.positionZ = savedCalibrationForRest.positionZ;
  calibration.rotationX = savedCalibrationForRest.rotationX;
  calibration.rotationY = savedCalibrationForRest.rotationY;
  calibration.rotationZ = savedCalibrationForRest.rotationZ;
  savedCalibrationForRest = null;
  updateSliderElementsFromCalibration();
  updateSliderLabels();
  applyRigTransform();
}

function updateModelVisibility() {
  // Hide all models first
  if (assetGroup) {
    assetGroup.visible = false;
  }
  if (assetGroupAlt) {
    assetGroupAlt.visible = false;
  }
  
  // Show the selected model
  if (renderMode === "asset" && assetLoaded && assetGroup) {
    assetGroup.visible = true;
  } else if (renderMode === "asset-alt" && assetLoadedAlt && assetGroupAlt) {
    assetGroupAlt.visible = true;
  }
}

function applyJointRigVisibility(forceVisible = false) {
  const shouldShow = forceVisible || overlayJointRig;
  setSynthSkeletonVisibility(shouldShow);
}

function initThree() {
  renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setClearColor(0x05060b, 1);

  scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x05060b, 8, 18);

  rigContainer = new THREE.Group();
  scene.add(rigContainer);

  jointGroup = new THREE.Group();
  rigContainer.add(jointGroup);

  const debugLineGeometry = new THREE.BufferGeometry();
  debugLineGeometry.setAttribute(
    "position",
    new THREE.BufferAttribute(debugLinePositions, 3),
  );
  const debugLineMaterial = new THREE.LineBasicMaterial({
    color: 0xffa54a,
    linewidth: 2,
  });
  debugLine = new THREE.Line(debugLineGeometry, debugLineMaterial);
  debugLine.visible = false;
  jointGroup.add(debugLine);

  camera = new THREE.PerspectiveCamera(
    45,
    window.innerWidth / window.innerHeight,
    0.1,
    100,
  );
  camera.position.set(0, 1.6, 4.5);
  camera.lookAt(0, 1.2, 0);

  const ambient = new THREE.AmbientLight(0xffffff, 0.45);
  scene.add(ambient);

  const keyLight = new THREE.DirectionalLight(0x78ffec, 1.15);
  keyLight.position.set(-2.5, 3.2, 2.5);
  scene.add(keyLight);

  const rimLight = new THREE.DirectionalLight(0x2f5cff, 0.75);
  rimLight.position.set(3, 2.8, -3);
  scene.add(rimLight);

  createSkeleton();
  loadSkeletonAsset("29human-skeleton.3DS", false);
  loadSkeletonAsset("alt-human-skeleton.3ds", true);
}

function createSkeleton() {
  const jointMaterial = new THREE.MeshStandardMaterial({
    color: 0xfff2a6,
    emissive: 0x332d11,
    roughness: 0.35,
    metalness: 0.15,
  });

  const boneMaterial = new THREE.MeshStandardMaterial({
    color: 0x6fd3ff,
    emissive: 0x08242f,
    roughness: 0.4,
    metalness: 0.2,
  });

  const jointGeometry = new THREE.SphereGeometry(0.03, 16, 16);
  const boneGeometry = new THREE.BoxGeometry(0.024, 1, 0.024);
  boneGeometry.translate(0, 0.5, 0);

  for (let i = 0; i < JOINT_COUNT; i += 1) {
    const joint = new THREE.Mesh(jointGeometry, jointMaterial);
    joint.visible = false;
    jointGroup.add(joint);
    joints.push(joint);
  }

  for (const [start, end] of BONE_PAIRS) {
    const bone = new THREE.Mesh(boneGeometry, boneMaterial);
    bone.visible = false;
    bone.castShadow = false;
    bone.receiveShadow = false;
    jointGroup.add(bone);
    bones.push({ mesh: bone, start, end });
  }
}

function loadSkeletonAsset(filename, isAlt) {
  const loader = new TDSLoader();
  loader.setResourcePath(".");
  loader.load(
    filename,
    (object) => {
      const group = new THREE.Group();
      group.visible = false;
      object.traverse((child) => {
        if (child.isMesh) {
          child.castShadow = false;
          child.receiveShadow = false;
          if (Array.isArray(child.material)) {
            child.material.forEach((material) => {
              if (material) {
                material.side = THREE.DoubleSide;
              }
            });
          } else if (child.material) {
            child.material.side = THREE.DoubleSide;
          }
        }
      });

      const bbox = new THREE.Box3().setFromObject(object);
      const size = new THREE.Vector3();
      bbox.getSize(size);
      const originalHeight = Math.max(size.y, 1e-3);
      const bboxCenter = new THREE.Vector3();
      bbox.getCenter(bboxCenter);
      const hipHeight = bbox.min.y + size.y * 0.53;
      const pivotOffset = new THREE.Vector3(bboxCenter.x, hipHeight, bboxCenter.z);
      const footOffset = Math.max(hipHeight - bbox.min.y, 1e-3);
      object.position.sub(pivotOffset);

      group.add(object);
      if (rigContainer) {
        rigContainer.add(group);
      } else {
        scene.add(group);
      }
      
      if (isAlt) {
        assetGroupAlt = group;
        assetOriginalHeightAlt = originalHeight;
        initializeAssetSegments(group, true);
        assetLoadedAlt = true;
        assetRestTransformAlt.position.copy(group.position);
        assetRestTransformAlt.quaternion.copy(group.quaternion);
        assetRestTransformAlt.scale.copy(group.scale);
        assetRestReadyAlt = true;
        console.info(
          `Alternate skeleton asset loaded. Original height: ${originalHeight.toFixed(3)}`,
        );
      } else {
        assetGroup = group;
        assetOriginalHeight = originalHeight;
        assetPivotOffset.copy(pivotOffset);
        assetFootOffset = footOffset;
        initializeAssetSegments(group, false);
        assetLoaded = true;
        assetRestTransform.position.copy(group.position);
        assetRestTransform.quaternion.copy(group.quaternion);
        assetRestTransform.scale.copy(group.scale);
        assetRestReady = true;
        console.info(
          `Skeleton asset loaded. Original height: ${originalHeight.toFixed(3)}`,
        );
        refreshSegmentAssignments();
        if (debugMode) {
          updateDebugUI();
        }
      }
      
      updateModelVisibility();
    },
    undefined,
    (error) => {
      console.error(`Failed to load ${isAlt ? 'alternate' : 'normal'} 3DS skeleton asset`, error);
      if (isAlt) {
        assetGroupAlt = null;
        assetLoadedAlt = false;
      } else {
        assetGroup = null;
        assetLoaded = false;
        renderMode = "none";
        renderModeSelect.value = "none";
        setStatus("3DS skeleton unavailable.", true);
      }
    },
  );
}

function initDebugControls() {
  if (!debugButton || !debugPanel) {
    return;
  }

  if (!landmarkSelectPopulated) {
    populateLandmarkSelect(debugStartSelect);
    populateLandmarkSelect(debugEndSelect);
    landmarkSelectPopulated = true;
  }

  debugButton.addEventListener("click", () => {
    if (!assetLoaded) {
      setStatus("Skeleton asset still loading. Try again shortly.", true);
      return;
    }
    openDebugPanel();
  });

  debugPrev.addEventListener("click", (event) => {
    event.preventDefault();
    if (!debugMode || assetSegments.length === 0) {
      return;
    }
    debugIndex = (debugIndex - 1 + assetSegments.length) % assetSegments.length;
    updateDebugUI();
  });

  debugNext.addEventListener("click", (event) => {
    event.preventDefault();
    if (!debugMode || assetSegments.length === 0) {
      return;
    }
    debugIndex = (debugIndex + 1) % assetSegments.length;
    updateDebugUI();
  });

  debugClose.addEventListener("click", (event) => {
    event.preventDefault();
    closeDebugPanel();
  });

  debugDump.addEventListener("click", (event) => {
    event.preventDefault();
    dumpDebugAnswers();
  });

  debugStartSelect.addEventListener("change", () => {
    if (!debugMode || assetSegments.length === 0) {
      return;
    }
    const segment = assetSegments[debugIndex];
    segment.userStart = parseInt(debugStartSelect.value, 10);
    if (Number.isNaN(segment.userStart)) {
      segment.userStart = segment.defaultStart;
    }
    refreshSegmentAssignments();
    updateDebugInfoPanel(segment);
  });

  debugEndSelect.addEventListener("change", () => {
    if (!debugMode || assetSegments.length === 0) {
      return;
    }
    const segment = assetSegments[debugIndex];
    segment.userEnd = parseInt(debugEndSelect.value, 10);
    if (Number.isNaN(segment.userEnd)) {
      segment.userEnd = segment.defaultEnd;
    }
    refreshSegmentAssignments();
    updateDebugInfoPanel(segment);
  });

  debugAxisSelect.addEventListener("change", () => {
    if (!debugMode || assetSegments.length === 0) {
      return;
    }
    const segment = assetSegments[debugIndex];
    segment.userAxisSelection = debugAxisSelect.value;
    refreshSegmentAssignments();
    updateDebugInfoPanel(segment);
  });

  debugNotes.addEventListener("input", () => {
    if (!debugMode || assetSegments.length === 0) {
      return;
    }
    const segment = assetSegments[debugIndex];
    segment.notes = debugNotes.value;
  });

  if (debugPreviewApply) {
    debugPreviewApply.addEventListener("change", () => {
      refreshSegmentAssignments();
    });
  }
}

function populateLandmarkSelect(select) {
  if (!select) {
    return;
  }
  select.innerHTML = "";
  for (const option of landmarkOptions) {
    const element = document.createElement("option");
    element.value = String(option.value);
    const pad = option.value.toString().padStart(2, "0");
    element.textContent = `${pad} — ${option.label}`;
    select.appendChild(element);
  }
}

function openDebugPanel() {
  if (!assetLoaded || assetSegments.every((segment) => !segment.mesh)) {
    setStatus("Skeleton asset not ready for debugging yet.", true);
    return;
  }
  debugMode = true;
  if (debugIndex >= assetSegments.length) {
    debugIndex = 0;
  }
  debugPanel.classList.add("visible");
  updateDebugUI();
}

function closeDebugPanel() {
  if (!debugMode) {
    return;
  }
  debugMode = false;
  debugPanel.classList.remove("visible");
  if (highlightedSegment) {
    setSegmentHighlight(highlightedSegment, false);
    highlightedSegment = null;
  }
  if (debugPreviewApply && debugPreviewApply.checked) {
    debugPreviewApply.checked = false;
    refreshSegmentAssignments();
  }
  if (debugLine) {
    debugLine.visible = false;
  }
}

function updateDebugUI() {
  if (!debugMode) {
    return;
  }
  if (assetSegments.length === 0) {
    debugInfo.textContent = "No mesh segments registered.";
    debugStepLabel.textContent = "";
    debugSegmentName.textContent = "";
    return;
  }
  const segment = assetSegments[debugIndex];
  debugStepLabel.textContent = `Segment ${debugIndex + 1} / ${assetSegments.length}`;
  debugSegmentName.textContent = segment.label
    ? `${segment.name} — ${segment.label}`
    : segment.name;
  debugStartSelect.value = String(segment.userStart ?? segment.defaultStart);
  debugEndSelect.value = String(segment.userEnd ?? segment.defaultEnd);
  debugAxisSelect.value = segment.userAxisSelection ?? "auto";
  debugNotes.value = segment.notes ?? "";
  updateDebugInfoPanel(segment);
  if (highlightedSegment && highlightedSegment !== segment) {
    setSegmentHighlight(highlightedSegment, false);
  }
  setSegmentHighlight(segment, true);
  highlightedSegment = segment;
  updateDebugGeometry(segment);
}

function updateDebugInfoPanel(segment) {
  const lines = [];
  lines.push(
    `Default mapping: ${formatLandmark(segment.defaultStart)} → ${formatLandmark(segment.defaultEnd)}`,
  );
  lines.push(
    `Selected mapping: ${formatLandmark(segment.userStart ?? segment.defaultStart)} → ${formatLandmark(segment.userEnd ?? segment.defaultEnd)}, ` +
      `${debugPreviewApply && debugPreviewApply.checked ? "applied" : "not applied"}`,
  );
  lines.push(
    `Axis (current): ${formatVector(segment.axis)} — selection: ${segment.userAxisSelection ?? "auto"}`,
  );
  lines.push(`Rest length: ${segment.restLength.toFixed(3)} units`);
  if (segment.mesh) {
    lines.push(
      `Local position (current frame): ${formatVector(segment.mesh.position)}`,
    );
    if (segment.restPosition) {
      lines.push(`Rest pivot (loaded): ${formatVector(segment.restPosition)}`);
    }
    if (segment.restBoundingBoxSize) {
      lines.push(
        `Rest bbox size: ${formatVector(segment.restBoundingBoxSize)}`,
      );
    }
  } else {
    lines.push("Mesh handle missing — confirm name in asset.");
  }
  const startIndex = segment.userStart ?? segment.defaultStart;
  const endIndex = segment.userEnd ?? segment.defaultEnd;
  if (
    typeof startIndex === "number" &&
    typeof endIndex === "number" &&
    jointVisibility[startIndex] &&
    jointVisibility[endIndex]
  ) {
    tempStart.copy(jointPositions[startIndex]);
    tempEnd.copy(jointPositions[endIndex]);
    tempDir.copy(tempEnd).sub(tempStart);
    lines.push(`Live joint delta: ${formatVector(tempDir)}`);
  } else {
    lines.push(
      "Live joint delta: unavailable (pose not detected for both joints)",
    );
  }
  if (segment.notes) {
    lines.push(`Notes: ${segment.notes}`);
  }
  lines.push(
    "Tip: enable Rest Pose to inspect the imported mesh alignment before mapping landmarks.",
  );
  debugInfo.textContent = lines.join("\n");
}

function setSegmentHighlight(segment, highlight) {
  if (!segment || !segment.mesh) {
    return;
  }
  const materials = Array.isArray(segment.mesh.material)
    ? segment.mesh.material
    : [segment.mesh.material];
  if (!segment.originalEmissiveArray) {
    segment.originalEmissiveArray = materials.map((material) =>
      material && material.emissive ? material.emissive.clone() : null,
    );
  }
  materials.forEach((material, index) => {
    if (!material || !material.emissive) {
      return;
    }
    const original = segment.originalEmissiveArray[index];
    if (highlight) {
      material.emissive.setRGB(0.5, 0.15, 0.05);
    } else if (original) {
      material.emissive.copy(original);
    }
  });
}

function updateDebugGeometry(segment) {
  if (!debugLine || !segment) {
    return;
  }
  if (!anyJointVisible()) {
    debugLine.visible = false;
    return;
  }
  const useApplied = Boolean(debugPreviewApply && debugPreviewApply.checked);
  const startIndex = useApplied
    ? segment.start
    : (segment.userStart ?? segment.defaultStart);
  const endIndex = useApplied
    ? segment.end
    : (segment.userEnd ?? segment.defaultEnd);
  if (
    !Number.isInteger(startIndex) ||
    !Number.isInteger(endIndex) ||
    startIndex < 0 ||
    endIndex < 0 ||
    startIndex >= JOINT_COUNT ||
    endIndex >= JOINT_COUNT
  ) {
    debugLine.visible = false;
    return;
  }
  if (!jointVisibility[startIndex] || !jointVisibility[endIndex]) {
    debugLine.visible = false;
    return;
  }
  if (!segment.mesh) {
    debugLine.visible = false;
    return;
  }
  const start = jointPositions[startIndex];
  const end = jointPositions[endIndex];
  debugLinePositions[0] = start.x;
  debugLinePositions[1] = start.y;
  debugLinePositions[2] = start.z;
  debugLinePositions[3] = end.x;
  debugLinePositions[4] = end.y;
  debugLinePositions[5] = end.z;
  const attribute = debugLine.geometry.getAttribute("position");
  attribute.needsUpdate = true;
  if (debugLine.geometry.boundingSphere) {
    debugLine.geometry.boundingSphere.center.set(0, 0, 0);
    debugLine.geometry.boundingSphere.radius = 0;
  }
  debugLine.geometry.computeBoundingSphere();
  debugLine.visible = true;
}

function refreshSegmentAssignments() {
  const applyPreview = Boolean(debugPreviewApply && debugPreviewApply.checked);
  for (const segment of assetSegments) {
    segment.start = applyPreview
      ? (segment.userStart ?? segment.defaultStart)
      : segment.defaultStart;
    segment.end = applyPreview
      ? (segment.userEnd ?? segment.defaultEnd)
      : segment.defaultEnd;
    updateSegmentAxisFromSelection(segment, applyPreview);
  }
  if (assetLoaded && assetGroup && anyJointVisible()) {
    assetGroup.updateMatrixWorld(true);
    updateAssetSegments();
  }
  if (debugMode && assetSegments.length > 0) {
    updateDebugGeometry(assetSegments[debugIndex]);
  }
}

function updateSegmentAxisFromSelection(segment, applyPreview) {
  const selection = segment.userAxisSelection ?? "auto";
  if (!applyPreview || selection === "auto") {
    segment.axis.copy(segment.defaultAxis);
    return;
  }
  switch (selection) {
    case "+x":
      segment.axis.set(1, 0, 0);
      break;
    case "-x":
      segment.axis.set(-1, 0, 0);
      break;
    case "+y":
      segment.axis.set(0, 1, 0);
      break;
    case "-y":
      segment.axis.set(0, -1, 0);
      break;
    case "+z":
      segment.axis.set(0, 0, 1);
      break;
    case "-z":
      segment.axis.set(0, 0, -1);
      break;
    default:
      segment.axis.copy(segment.defaultAxis);
      break;
  }
  segment.axis.normalize();
}

function dumpDebugAnswers() {
  const payload = assetSegments.map((segment) => ({
    mesh: segment.name,
    label: segment.label,
    defaultStart: segment.defaultStart,
    defaultEnd: segment.defaultEnd,
    selectedStart: segment.userStart,
    selectedEnd: segment.userEnd,
    axis: segment.userAxisSelection,
    notes: segment.notes,
  }));
  console.groupCollapsed("Mesh debug answers");
  console.table(payload);
  console.groupEnd();
  setStatus("Exported debug data to console.");
}

function anyJointVisible() {
  return jointVisibility.some((value) => value === true);
}

function formatLandmark(index) {
  if (typeof index !== "number" || Number.isNaN(index)) {
    return "(unset)";
  }
  const label = landmarkOptions[index]?.label ?? "Unknown";
  return `${index} ${label}`;
}

function formatVector(vector) {
  if (!vector) {
    return "[?, ?, ?]";
  }
  return `[${vector.x.toFixed(3)}, ${vector.y.toFixed(3)}, ${vector.z.toFixed(3)}]`;
}

function initializeAssetSegments(root, isAlt) {
  root.updateMatrixWorld(true);
  for (const segment of assetSegments) {
    const mesh = root.getObjectByName(segment.name);
    if (!mesh) {
      console.warn(`Segment mesh "${segment.name}" not found in ${isAlt ? 'alternate' : 'normal'} 3DS asset.`);
      if (isAlt) {
        segment.meshAlt = null;
      } else {
        segment.mesh = null;
      }
      continue;
    }
    if (Array.isArray(mesh.material)) {
      mesh.material = mesh.material.map((mat) => (mat ? mat.clone() : mat));
    } else if (mesh.material) {
      mesh.material = mesh.material.clone();
    }
    if (!mesh.geometry.boundingBox) {
      mesh.geometry.computeBoundingBox();
    }
    
    if (isAlt) {
      segment.meshAlt = mesh;
    } else {
      segment.mesh = mesh;
    }
    
    mesh.matrixAutoUpdate = true;
    
    const baseScale = isAlt ? segment.baseScaleAlt : segment.baseScale;
    const initialQuaternion = isAlt ? segment.initialQuaternionAlt : segment.initialQuaternion;
    const restPosition = isAlt ? segment.restPositionAlt : segment.restPosition;
    
    baseScale.copy(mesh.scale);
    initialQuaternion.copy(mesh.quaternion);
    restPosition.copy(mesh.position);
    
    const materials = Array.isArray(mesh.material)
      ? mesh.material
      : [mesh.material];
    
    if (isAlt) {
      segment.originalEmissiveArrayAlt = materials.map((material) =>
        material && material.emissive ? material.emissive.clone() : null,
      );
    } else {
      segment.originalEmissiveArray = materials.map((material) =>
        material && material.emissive ? material.emissive.clone() : null,
      );
    }

    // Get bounding box in LOCAL space (before transforms)
    const localBBox = mesh.geometry.boundingBox.clone();
    const localSize = new THREE.Vector3();
    localBBox.getSize(localSize);
    const localCenter = new THREE.Vector3();
    localBBox.getCenter(localCenter);

    // Center the geometry around its local origin
    mesh.geometry.translate(-localCenter.x, -localCenter.y, -localCenter.z);
    
    // Apply coordinate system transform directly to geometry (Z-up to Y-up)
    const rotMatrix = new THREE.Matrix4().makeRotationFromQuaternion(coordSystemTransform);
    mesh.geometry.applyMatrix4(rotMatrix);
    
    // Apply additional 180-degree rotation around Y to correct palm orientation
    const flipMatrix = new THREE.Matrix4().makeRotationY(Math.PI);
    mesh.geometry.applyMatrix4(flipMatrix);
    
    // Apply part-specific corrections
    if (segment.name === "object_7") {
      // Neck - rotate 180° on Y axis to face forward
      const neckFlip = new THREE.Matrix4().makeRotationY(Math.PI);
      mesh.geometry.applyMatrix4(neckFlip);
    } else if (segment.name === "object_8") {
      // Skull - rotate 180° on Y axis to face forward, then 180° on X to flip right-side up
      const skullFlipY = new THREE.Matrix4().makeRotationY(Math.PI);
      mesh.geometry.applyMatrix4(skullFlipY);
      const skullFlipX = new THREE.Matrix4().makeRotationX(Math.PI);
      mesh.geometry.applyMatrix4(skullFlipX);
    } else if (segment.name === "object_9") {
      // Jaw - rotate 180° on Y axis to face forward
      const jawFlip = new THREE.Matrix4().makeRotationY(Math.PI);
      mesh.geometry.applyMatrix4(jawFlip);
    } else if (segment.name === "object_3") {
      // Chest - rotate 180° on Y axis to face forward
      const chestFlipY = new THREE.Matrix4().makeRotationY(Math.PI);
      mesh.geometry.applyMatrix4(chestFlipY);
    }
    
    mesh.geometry.computeBoundingBox();
    
    // Recompute bbox after centering and rotating
    const centeredBBox = mesh.geometry.boundingBox.clone();
    centeredBBox.getSize(localSize);

    // Determine primary axis based on size
    let axisIndex = 1;
    let length = localSize.y;
    if (localSize.x >= localSize.y && localSize.x >= localSize.z) {
      axisIndex = 0;
      length = localSize.x;
    } else if (localSize.z >= localSize.x && localSize.z >= localSize.y) {
      axisIndex = 2;
      length = localSize.z;
    }

    // Set the axis direction (already in Y-up space since geometry was transformed)
    const axis = isAlt ? segment.axisAlt : segment.axis;
    const defaultAxis = isAlt ? segment.defaultAxisAlt : segment.defaultAxis;
    
    if (axisIndex === 0) {
      axis.set(1, 0, 0);
    } else if (axisIndex === 1) {
      axis.set(0, 1, 0);
    } else {
      axis.set(0, 0, 1);
    }
    axis.normalize();
    
    if (isAlt) {
      segment.restLengthAlt = Math.max(length, 1e-3);
    } else {
      segment.restLength = Math.max(length, 1e-3);
    }
    defaultAxis.copy(axis);
    
    const bbox = new THREE.Box3().setFromObject(mesh);
    if (isAlt) {
      segment.restBoundingBoxAlt = bbox.clone();
      segment.restBoundingBoxSizeAlt = localSize.clone();
    } else {
      segment.restBoundingBox = bbox.clone();
      segment.restBoundingBoxSize = localSize.clone();
    }
  }
}

function animate() {
  animationFrameId = requestAnimationFrame(animate);

  applyRigTransform();

  if (!poseTrackingEnabled) {
    applyRestPose();
    renderer.render(scene, camera);
    return;
  }

  if (
    !poseLandmarker ||
    videoElement.readyState < HTMLMediaElement.HAVE_CURRENT_DATA
  ) {
    renderer.render(scene, camera);
    return;
  }

  if (lastVideoTime === videoElement.currentTime) {
    renderer.render(scene, camera);
    return;
  }

  lastVideoTime = videoElement.currentTime;
  const results = poseLandmarker.detectForVideo(
    videoElement,
    performance.now(),
  );

  if (results && results.landmarks && results.landmarks.length > 0) {
    updateSkeleton(results.landmarks[0]);
  } else {
    handleNoPose();
  }

  if (debugMode && assetSegments.length > 0) {
    updateDebugGeometry(assetSegments[debugIndex]);
  }

  renderer.render(scene, camera);
}

function updateSkeleton(landmarks) {
  computeJointData(landmarks);
  updateSynthSkeleton();

  // Update asset pose based on selected model
  if (renderMode === "asset" && assetLoaded) {
    updateAssetPose(false);
  } else if (renderMode === "asset-alt" && assetLoadedAlt) {
    updateAssetPose(true);
  }
  
  updateModelVisibility();
  applyJointRigVisibility();

  if (debugMode && assetSegments.length > 0) {
    updateDebugGeometry(assetSegments[debugIndex]);
  }
}

function computeJointData(landmarks) {
  const smoothing = THREE.MathUtils.clamp(calibration.smoothing, 0, 0.95);
  const lerpAlpha = 1 - smoothing;

  for (let i = 0; i < JOINT_COUNT; i += 1) {
    const lm = landmarks[i];
    const wasVisible = jointVisibility[i];
    if (!lm) {
      jointVisibility[i] = false;
      continue;
    }
    jointVisibility[i] = true;
    const targetX = calibration.mirror
      ? (0.5 - lm.x) * calibration.scaleX
      : (lm.x - 0.5) * calibration.scaleX;
    const targetY = (0.5 - lm.y) * calibration.scaleY;
    const targetZ = -lm.z * calibration.scaleZ;
    if (!wasVisible || lerpAlpha >= 1) {
      jointPositions[i].set(targetX, targetY, targetZ);
    } else if (lerpAlpha <= 0) {
      // Keep previous value when smoothing is maximal.
    } else {
      const jp = jointPositions[i];
      jp.x += (targetX - jp.x) * lerpAlpha;
      jp.y += (targetY - jp.y) * lerpAlpha;
      jp.z += (targetZ - jp.z) * lerpAlpha;
    }
  }
}

function calculateBoneRotation(startIdx, endIdx, direction, outQuaternion) {
  const configKey = `${startIdx}-${endIdx}`;
  const config = BONE_ROTATION_CONFIG[configKey];
  
  if (boneRotationStrategy === "unconstrained" || !config) {
    // Strategy 1: Unconstrained - just align Y axis to direction
    outQuaternion.setFromUnitVectors(yAxis, direction);
    return;
  }
  
  if (boneRotationStrategy === "pole") {
    // Strategy 2: Three-joint pole vector
    const poleIdx = config.pole;
    if (!jointVisibility[poleIdx]) {
      // Fall back to unconstrained if pole joint not visible
      outQuaternion.setFromUnitVectors(yAxis, direction);
      return;
    }
    
    // Calculate pole vector (from start towards pole joint)
    const startPos = jointPositions[startIdx];
    const polePos = jointPositions[poleIdx];
    tempPoleVec.copy(polePos).sub(startPos);
    
    // Remove component along bone direction to get perpendicular component
    const projection = tempPoleVec.dot(direction);
    tempPoleVec.addScaledVector(direction, -projection);
    
    // Check if pole vector is too small (parallel to bone)
    if (tempPoleVec.lengthSq() < 1e-6) {
      outQuaternion.setFromUnitVectors(yAxis, direction);
      return;
    }
    tempPoleVec.normalize();
    
    // Calculate the right vector (perpendicular to both direction and pole)
    tempRight.crossVectors(direction, tempPoleVec).normalize();
    
    // Recalculate the forward vector to be perfectly perpendicular
    tempUpVec.crossVectors(tempRight, direction).normalize();
    
    // Build rotation matrix from orthonormal basis (X=right, Y=up/direction, Z=forward)
    tempMatrix.makeBasis(tempRight, direction, tempUpVec);
    outQuaternion.setFromRotationMatrix(tempMatrix);
    return;
  }
  
  if (boneRotationStrategy === "anatomical") {
    // Strategy 3: Anatomical plane with predefined up direction
    const anatomicalUp = config.anatomical.clone();
    
    // Remove component along bone direction
    const projection = anatomicalUp.dot(direction);
    anatomicalUp.addScaledVector(direction, -projection);
    
    if (anatomicalUp.lengthSq() < 1e-6) {
      outQuaternion.setFromUnitVectors(yAxis, direction);
      return;
    }
    anatomicalUp.normalize();
    
    // Calculate the right vector
    tempRight.crossVectors(direction, anatomicalUp).normalize();
    
    // Recalculate the forward vector to be perfectly perpendicular
    tempUpVec.crossVectors(tempRight, direction).normalize();
    
    // Build rotation matrix from orthonormal basis
    tempMatrix.makeBasis(tempRight, direction, tempUpVec);
    outQuaternion.setFromRotationMatrix(tempMatrix);
    return;
  }
}

function calculateAssetRotation(segment, startIdx, endIdx, direction, outQuaternion, axis, initialQuaternion) {
  // Transform the segment's natural axis by its initial rotation
  const meshAxis = axis.clone().applyQuaternion(initialQuaternion);
  
  const configKey = `${startIdx}-${endIdx}`;
  const config = BONE_ROTATION_CONFIG[configKey];
  
  if (boneRotationStrategy === "unconstrained" || !config) {
    // Strategy 1: Unconstrained - just align mesh axis to direction
    tempAlignQuat.setFromUnitVectors(meshAxis, direction);
    outQuaternion.copy(initialQuaternion).premultiply(tempAlignQuat);
    return;
  }
  
  if (boneRotationStrategy === "pole") {
    // Strategy 2: Three-joint pole vector
    const poleIdx = config.pole;
    if (!jointVisibility[poleIdx]) {
      // Fall back to unconstrained
      tempAlignQuat.setFromUnitVectors(meshAxis, direction);
      outQuaternion.copy(initialQuaternion).premultiply(tempAlignQuat);
      return;
    }
    
    // Calculate pole vector (from start towards pole joint)
    const startPos = jointPositions[startIdx];
    const polePos = jointPositions[poleIdx];
    tempPoleVec.copy(polePos).sub(startPos);
    
    // Remove component along bone direction
    const projection = tempPoleVec.dot(direction);
    tempPoleVec.addScaledVector(direction, -projection);
    
    if (tempPoleVec.lengthSq() < 1e-6) {
      tempAlignQuat.setFromUnitVectors(meshAxis, direction);
      outQuaternion.copy(initialQuaternion).premultiply(tempAlignQuat);
      return;
    }
    tempPoleVec.normalize();
    
    // Build target frame with pole constraint
    tempRight.crossVectors(direction, tempPoleVec).normalize();
    tempUpVec.crossVectors(tempRight, direction).normalize();
    tempMatrix.makeBasis(tempRight, direction, tempUpVec);
    
    // Create quaternion that rotates mesh axis to this frame
    const targetQuat = new THREE.Quaternion().setFromRotationMatrix(tempMatrix);
    
    // Find rotation from mesh axis to Y axis, then to target
    const meshToY = new THREE.Quaternion().setFromUnitVectors(meshAxis, yAxis);
    outQuaternion.copy(initialQuaternion).premultiply(meshToY).premultiply(targetQuat);
    return;
  }
  
  if (boneRotationStrategy === "anatomical") {
    // Strategy 3: Anatomical plane with predefined up direction
    const anatomicalUp = config.anatomical.clone();
    
    // Remove component along bone direction
    const projection = anatomicalUp.dot(direction);
    anatomicalUp.addScaledVector(direction, -projection);
    
    if (anatomicalUp.lengthSq() < 1e-6) {
      tempAlignQuat.setFromUnitVectors(meshAxis, direction);
      outQuaternion.copy(initialQuaternion).premultiply(tempAlignQuat);
      return;
    }
    anatomicalUp.normalize();
    
    // Build target frame with anatomical constraint
    tempRight.crossVectors(direction, anatomicalUp).normalize();
    tempUpVec.crossVectors(tempRight, direction).normalize();
    tempMatrix.makeBasis(tempRight, direction, tempUpVec);
    
    // Create quaternion that rotates mesh axis to this frame
    const targetQuat = new THREE.Quaternion().setFromRotationMatrix(tempMatrix);
    
    // Find rotation from mesh axis to Y axis, then to target
    const meshToY = new THREE.Quaternion().setFromUnitVectors(meshAxis, yAxis);
    outQuaternion.copy(initialQuaternion).premultiply(meshToY).premultiply(targetQuat);
    return;
  }
}

function updateSynthSkeleton() {
  for (let i = 0; i < JOINT_COUNT; i += 1) {
    if (!jointVisibility[i]) {
      joints[i].visible = false;
      continue;
    }
    joints[i].position.copy(jointPositions[i]);
    joints[i].visible = true;
  }
  for (const bone of bones) {
    const startPos = jointPositions[bone.start];
    const endPos = jointPositions[bone.end];
    if (!jointVisibility[bone.start] || !jointVisibility[bone.end]) {
      bone.mesh.visible = false;
      continue;
    }
    tempStart.copy(startPos);
    tempEnd.copy(endPos);
    tempDir.copy(tempEnd).sub(tempStart);
    const length = tempDir.length();
    if (length === 0) {
      bone.mesh.visible = false;
      continue;
    }
    bone.mesh.position.copy(tempStart).addScaledVector(tempDir, 0.5);
    bone.mesh.scale.set(1, length, 1);
    
    // Calculate rotation based on selected strategy
    tempDir.normalize();
    calculateBoneRotation(bone.start, bone.end, tempDir, bone.mesh.quaternion);
    
    bone.mesh.visible = true;
  }
}

function updateAssetPose(isAlt) {
  const group = isAlt ? assetGroupAlt : assetGroup;
  const loaded = isAlt ? assetLoadedAlt : assetLoaded;
  
  if (!loaded || !group) {
    return false;
  }
  if (rigContainer) {
    rigContainer.updateMatrixWorld(true);
  }
  group.position.set(0, 0, 0);
  group.quaternion.identity();
  group.scale.set(1, 1, 1);
  group.updateMatrixWorld(true);
  updateAssetSegments(isAlt);
  setAssetVisibility(true);
  return true;
}

function updateAssetSegments(isAlt) {
  const group = isAlt ? assetGroupAlt : assetGroup;
  const loaded = isAlt ? assetLoadedAlt : assetLoaded;
  
  if (!loaded || !group) {
    return;
  }
  for (const segment of assetSegments) {
    const mesh = isAlt ? segment.meshAlt : segment.mesh;
    if (!mesh) {
      continue;
    }
    
    // Get the appropriate properties based on which model
    const baseScale = isAlt ? segment.baseScaleAlt : segment.baseScale;
    const initialQuaternion = isAlt ? segment.initialQuaternionAlt : segment.initialQuaternion;
    const restPosition = isAlt ? segment.restPositionAlt : segment.restPosition;
    const axis = isAlt ? segment.axisAlt : segment.axis;
    
    const isTargetSegment = animatedSegmentNames.has(segment.name);

    if (!isTargetSegment) {
      mesh.position.copy(restPosition);
      mesh.quaternion.copy(initialQuaternion);
      mesh.scale.copy(baseScale);
      mesh.visible = true;
      continue;
    }

    if (!jointVisibility[segment.start] || !jointVisibility[segment.end]) {
      mesh.position.copy(restPosition);
      mesh.quaternion.copy(initialQuaternion);
      mesh.scale.copy(baseScale);
      mesh.visible = false;
      continue;
    }

    const startLocal = jointPositions[segment.start];
    const endLocal = jointPositions[segment.end];

    tempDirLocal.copy(endLocal).sub(startLocal);
    const length = tempDirLocal.length();
    
    // Handle parts with same start/end landmark (like skull, jaw, shoulders)
    if (length < 1e-4) {
      // Just position at the landmark with initial rotation
      mesh.position.copy(startLocal);
      mesh.quaternion.copy(initialQuaternion);
      mesh.scale.copy(baseScale);
      mesh.visible = true;
      ensureSegmentDebugSphere(segment, mesh, isAlt);
      continue;
    }
    
    tempDirLocal.normalize();
    
    // Calculate rotation based on selected strategy
    calculateAssetRotation(segment, segment.start, segment.end, tempDirLocal, mesh.quaternion, axis, initialQuaternion);
    
    // Position mesh at the midpoint of the limb (since geometry is now centered)
    mesh.position.copy(startLocal).add(tempDirLocal.clone().multiplyScalar(length / 2));
    
    // Keep original scale
    mesh.scale.copy(baseScale);
    mesh.visible = true;

    ensureSegmentDebugSphere(segment, mesh, isAlt);
  }
}

function ensureSegmentDebugSphere(segment, mesh, isAlt) {
  const group = isAlt ? assetGroupAlt : assetGroup;
  if (!group) {
    return;
  }
  const key = isAlt ? `${segment.name}-alt` : segment.name;
  let sphere = segmentDebugSpheres.get(key);
  if (!sphere) {
    const geometry = new THREE.SphereGeometry(0.04, 16, 16);
    const material = new THREE.MeshBasicMaterial({ color: 0xffa54a });
    sphere = new THREE.Mesh(geometry, material);
    sphere.renderOrder = 10;
    mesh.add(sphere);
    segmentDebugSpheres.set(key, sphere);
  }
  sphere.position.set(0, 0, 0);
  sphere.scale.set(
    1 / mesh.scale.x,
    1 / mesh.scale.y,
    1 / mesh.scale.z
  );
  sphere.visible = mesh.visible && showDebugSpheres;
}

function updateDebugSpheresVisibility() {
  for (const [key, sphere] of segmentDebugSpheres) {
    if (sphere.parent && sphere.parent.visible) {
      sphere.visible = showDebugSpheres;
    } else {
      sphere.visible = false;
    }
  }
}

function applyRigTransform() {
  if (!rigContainer) {
    return;
  }
  rigContainer.position.set(
    calibration.positionX,
    calibration.positionY,
    calibration.positionZ,
  );
  const rx = THREE.MathUtils.degToRad(calibration.rotationX);
  const ry = THREE.MathUtils.degToRad(calibration.rotationY);
  const rz = THREE.MathUtils.degToRad(calibration.rotationZ);

  const quat = new THREE.Quaternion();
  const qy = new THREE.Quaternion().setFromAxisAngle(yAxis, ry);
  const qx = new THREE.Quaternion().setFromAxisAngle(xAxis, rx);
  const qz = new THREE.Quaternion().setFromAxisAngle(zAxis, rz);

  // Apply rotations in YXZ order to minimize gimbal coupling.
  quat.multiply(qy).multiply(qx).multiply(qz);
  rigContainer.quaternion.copy(quat);
}

function applyRestPose() {
  clearJointVisibility();
  if (renderMode === "asset" && assetGroup && assetLoaded) {
    if (assetRestReady) {
      assetGroup.position.copy(assetRestTransform.position);
      assetGroup.quaternion.copy(assetRestTransform.quaternion);
      assetGroup.scale.copy(assetRestTransform.scale);
    }
    resetSegmentsToRest();
    setAssetVisibility(true);
  } else if (renderMode === "asset-alt" && assetGroupAlt && assetLoadedAlt) {
    if (assetRestReadyAlt) {
      assetGroupAlt.position.copy(assetRestTransformAlt.position);
      assetGroupAlt.quaternion.copy(assetRestTransformAlt.quaternion);
      assetGroupAlt.scale.copy(assetRestTransformAlt.scale);
    }
    resetSegmentsToRest();
    setAssetVisibility(true);
  } else {
    setAssetVisibility(false);
  }
  applyJointRigVisibility();
  if (debugLine) {
    debugLine.visible = false;
  }
}

function resetSegmentsToRest() {
  const isAlt = renderMode === "asset-alt";
  const loaded = isAlt ? assetLoadedAlt : assetLoaded;
  
  if (!loaded) {
    return;
  }
  for (const segment of assetSegments) {
    const mesh = isAlt ? segment.meshAlt : segment.mesh;
    if (!mesh) {
      continue;
    }
    const restPosition = isAlt ? segment.restPositionAlt : segment.restPosition;
    const initialQuaternion = isAlt ? segment.initialQuaternionAlt : segment.initialQuaternion;
    const baseScale = isAlt ? segment.baseScaleAlt : segment.baseScale;
    
    mesh.position.copy(restPosition);
    mesh.quaternion.copy(initialQuaternion);
    mesh.scale.copy(baseScale);
    mesh.visible = true;
  }
}

function setSynthSkeletonVisibility(visible) {
  for (let i = 0; i < JOINT_COUNT; i += 1) {
    joints[i].visible = visible && jointVisibility[i];
  }
  for (const bone of bones) {
    bone.mesh.visible =
      visible && jointVisibility[bone.start] && jointVisibility[bone.end];
  }
}

function setAssetVisibility(visible) {
  if (visible) {
    updateModelVisibility();
  } else {
    if (assetGroup) {
      assetGroup.visible = false;
    }
    if (assetGroupAlt) {
      assetGroupAlt.visible = false;
    }
  }
}

function clearJointVisibility() {
  for (let i = 0; i < JOINT_COUNT; i += 1) {
    jointVisibility[i] = false;
  }
}

function handleNoPose() {
  clearJointVisibility();
  if (renderMode !== "none") {
    setAssetVisibility(false);
  }
  applyJointRigVisibility();
  if (debugLine) {
    debugLine.visible = false;
  }
}

function setStatus(message, isError = false) {
  statusLabel.textContent = message;
  statusLabel.style.color = isError ? "#ff8a8a" : "rgba(255,255,255,0.7)";
}

window.addEventListener("unload", () => {
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
  }
  if (poseLandmarker) {
    poseLandmarker.close();
  }
  if (videoElement && videoElement.srcObject) {
    const tracks = videoElement.srcObject.getTracks();
    tracks.forEach((track) => track.stop());
  }
});
