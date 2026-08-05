import 'package:flutter/material.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:camera/camera.dart';
import 'pose_model.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Wall Ball Form App',
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final PoseModel _poseModel = PoseModel();
  bool _hasAccelerometer = false;
  bool _hasFrontCamera = false;
  bool _hasRearCamera = false;
  bool _modelLoaded = false;
  bool _checked = false;

  @override
  void initState() {
    super.initState();
    _checkHardware();
    _loadModel();
  }

  @override
  void dispose() {
    _poseModel.dispose();
    super.dispose();
  }

  Future<void> _loadModel() async {
    await _poseModel.initialize();
    if (mounted) {
      setState(() {
        _modelLoaded = _poseModel.isLoaded;
      });
    }
  }

  Future<void> _checkHardware() async {
    bool accelAvailable = false;
    try {
      await accelerometerEvents.first.timeout(const Duration(milliseconds: 500));
      accelAvailable = true;
    } catch (_) {
      accelAvailable = false;
    }

    List<CameraDescription> cameras = [];
    try {
      cameras = await availableCameras();
    } catch (_) {
      cameras = [];
    }

    bool front = cameras.any((c) => c.lensDirection == CameraLensDirection.front);
    bool rear = cameras.any((c) => c.lensDirection == CameraLensDirection.back);

    if (mounted) {
      setState(() {
        _hasAccelerometer = accelAvailable;
        _hasFrontCamera = front;
        _hasRearCamera = rear;
        _checked = true;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Hello World')),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Text('Hello, world!', style: TextStyle(fontSize: 24)),
            const SizedBox(height: 24),
            if (!_checked || _poseModel.isLoading)
              const CircularProgressIndicator()
            else
              Column(
                children: [
                  _HardwareRow(
                    label: 'IMU / Accelerometer',
                    available: _hasAccelerometer,
                  ),
                  _HardwareRow(
                    label: 'Front Camera',
                    available: _hasFrontCamera,
                  ),
                  _HardwareRow(
                    label: 'Rear Camera',
                    available: _hasRearCamera,
                  ),
                  _HardwareRow(
                    label: 'Pose Model (ONNX)',
                    available: _modelLoaded,
                    error: _poseModel.loadError?.toString(),
                  ),
                ],
              ),
          ],
        ),
      ),
    );
  }
}

class _HardwareRow extends StatelessWidget {
  final String label;
  final bool available;
  final String? error;

  const _HardwareRow({required this.label, required this.available, this.error});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(
                available ? Icons.check_circle : Icons.cancel,
                color: available ? Colors.green : Colors.red,
                size: 24,
              ),
              const SizedBox(width: 12),
              Text(
                '$label: ${available ? "Yes" : "No"}',
                style: TextStyle(
                  fontSize: 16,
                  color: available ? Colors.green : Colors.red,
                ),
              ),
            ],
          ),
          if (error != null && !available)
            Padding(
              padding: const EdgeInsets.only(left: 36, top: 4),
              child: Text(
                error!,
                style: const TextStyle(fontSize: 12, color: Colors.red),
              ),
            ),
        ],
      ),
    );
  }
}