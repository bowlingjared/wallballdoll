import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:ort/ort.dart';
import 'package:path_provider/path_provider.dart';

class PoseModel {
  static const String _assetPath = 'assets/model/pose_model.onnx';
  static const String _fileName = 'pose_model.onnx';

  Session? _session;
  String? _modelPath;
  bool _isLoading = false;
  bool _isLoaded = false;
  Object? _loadError;

  bool get isLoaded => _isLoaded;
  bool get isLoading => _isLoading;
  Object? get loadError => _loadError;
  Session? get session => _session;

  Future<void> initialize() async {
    if (_isLoaded || _isLoading) return;
    _isLoading = true;
    _loadError = null;

    try {
      // Initialize ORT runtime
      final initialized = await Ort.ensureInitialized(
        options: OrtInitializationOptions(
          showDebugMessages: true,
        ),
      );
      if (!initialized) {
        throw Exception('Failed to initialize ORT');
      }

      _modelPath = await _copyModelToFile();
      _session = await Session.builder().commitFromFile(_modelPath!);
      _isLoaded = true;
      print('[PoseModel] Model loaded successfully from $_modelPath');
      print('[PoseModel] Inputs: ${_session!.inputNames.toList()}');
      print('[PoseModel] Outputs: ${_session!.outputNames.toList()}');
    } catch (e, st) {
      _loadError = e;
      print('[PoseModel] Failed to load model: $e');
      print(st);
    } finally {
      _isLoading = false;
    }
  }

  Future<String> _copyModelToFile() async {
    final dir = await getApplicationDocumentsDirectory();
    final file = File('${dir.path}/$_fileName');
    
    if (await file.exists()) {
      return file.path;
    }

    final data = await rootBundle.load(_assetPath);
    await file.writeAsBytes(data.buffer.asUint8List());
    return file.path;
  }

  /// Run inference on preprocessed image tensor [1, 3, 256, 192]
  /// Returns [simcc_x, simcc_y] as Float32Lists
  Future<List<Float32List>?> run(Uint8List rgbBytes, int width, int height) async {
    if (_session == null || !_isLoaded) return null;

    try {
      // Preprocess: resize to 256x192, normalize to float32 [0,1]
      final inputTensor = _preprocessImage(rgbBytes, width, height);
      
      final inputs = {'input': inputTensor};
      final outputs = await _session!.run(inputValues: inputs);
      
      final simccX = outputs['simcc_x'] as Tensor<double>;
      final simccY = outputs['simcc_y'] as Tensor<double>;
      
      return [
        Float32List.fromList(simccX.data),
        Float32List.fromList(simccY.data),
      ];
    } catch (e) {
      print('[PoseModel] Inference error: $e');
      return null;
    }
  }

  Tensor<double> _preprocessImage(Uint8List rgbBytes, int width, int height) {
    // Resize and normalize to 256x192, NCHW format
    const targetW = 256;
    const targetH = 192;
    const channels = 3;
    
    final floatData = Float32List(targetW * targetH * channels);
    
    // Simple nearest-neighbor resize + normalize to [0,1]
    final scaleX = width / targetW;
    final scaleY = height / targetH;
    
    for (int c = 0; c < channels; c++) {
      for (int y = 0; y < targetH; y++) {
        for (int x = 0; x < targetW; x++) {
          final srcX = (x * scaleX).clamp(0, width - 1).toInt();
          final srcY = (y * scaleY).clamp(0, height - 1).toInt();
          final srcIdx = (srcY * width + srcX) * channels + c;
          final dstIdx = (c * targetH + y) * targetW + x;
          floatData[dstIdx] = rgbBytes[srcIdx] / 255.0;
        }
      }
    }
    
    return Tensor.fromArrayF32(shape: [1, channels, targetH, targetW], data: floatData.toList());
  }

  /// Convert SimCC outputs to keypoint coordinates
  /// simcc_x: [1, 133, 384], simcc_y: [1, 133, 512]
  /// Returns list of 133 (x,y) pairs in original image coordinates
  List<Point>? decodeSimcc(Float32List simccX, Float32List simccY, 
                           int origWidth, int origHeight) {
    const kpts = 133;
    const simccW = 384;
    const simccH = 512;
    const inputW = 256;
    const inputH = 192;
    
    final points = <Point>[];
    
    for (int i = 0; i < kpts; i++) {
      // Find max in simcc_x[i] (length 384)
      int maxXIdx = 0;
      double maxXVal = simccX[i * simccW];
      for (int j = 1; j < simccW; j++) {
        final val = simccX[i * simccW + j];
        if (val > maxXVal) {
          maxXVal = val;
          maxXIdx = j;
        }
      }
      
      // Find max in simcc_y[i] (length 512)
      int maxYIdx = 0;
      double maxYVal = simccY[i * simccH];
      for (int j = 1; j < simccH; j++) {
        final val = simccY[i * simccH + j];
        if (val > maxYVal) {
          maxYVal = val;
          maxYIdx = j;
        }
      }
      
      // Map from SimCC space (384x512) -> input space (256x192) -> original
      final x = (maxXIdx / simccW) * inputW * (origWidth / inputW);
      final y = (maxYIdx / simccH) * inputH * (origHeight / inputH);
      
      points.add(Point(x, y, maxXVal * maxYVal));
    }
    
    return points;
  }

  void dispose() {
    _session?.dispose();
    _session = null;
    _isLoaded = false;
  }
}

class Point {
  final double x;
  final double y;
  final double confidence;
  
  Point(this.x, this.y, this.confidence);
}