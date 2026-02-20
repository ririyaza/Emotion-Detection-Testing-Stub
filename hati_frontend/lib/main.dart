import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:record/record.dart';
import 'dart:convert';
import 'dart:io';
import 'package:path_provider/path_provider.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      debugShowCheckedModeBanner: false,
      home: EmotionPage(),
    );
  }
}

class EmotionPage extends StatefulWidget {
  const EmotionPage({super.key});

  @override
  State<EmotionPage> createState() => _EmotionPageState();
}

class _EmotionPageState extends State<EmotionPage> {
  final TextEditingController _controller = TextEditingController();
  final AudioRecorder _record = AudioRecorder();

  String emotionResult = "No result yet";
  bool isRecording = false;
  bool isLoading = false;
  String baseUrl = "http://10.102.135.110:5000";

  Future<void> predictText() async {
    setState(() => isLoading = true);

    try {
      final response = await http.post(
        Uri.parse("$baseUrl/predict_text"),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({"text": _controller.text}),
      );

      final data = jsonDecode(response.body);

      setState(() {
        emotionResult = data["emotion"].toString();
      });
    } catch (e) {
      setState(() {
        emotionResult = "Error: $e";
      });
    } finally {
      setState(() => isLoading = false);
    }
  }

  Future<void> startRecording() async {
    if (await _record.hasPermission()) {
      final dir = await getTemporaryDirectory();
      final path = '${dir.path}/record.wav';

      await _record.start(
        const RecordConfig(
          encoder: AudioEncoder.wav,
          sampleRate: 44100,
          bitRate: 128000,
        ),
        path: path,
      );

      setState(() {
        isRecording = true;
      });
    } else {
      setState(() {
        emotionResult = "Microphone permission denied";
      });
    }
  }

  Future<void> stopRecording() async {
    setState(() => isLoading = true);

    final path = await _record.stop();

    setState(() {
      isRecording = false;
    });

    if (path != null) {
      await sendAudio(File(path));
    }

    setState(() => isLoading = false);
  }

  Future<void> sendAudio(File file) async {
    try {
      var request = http.MultipartRequest(
        "POST",
        Uri.parse("$baseUrl/predict_audio"),
      );

      request.files.add(await http.MultipartFile.fromPath("audio", file.path));

      var response = await request.send();
      var responseData = await response.stream.bytesToString();
      var data = jsonDecode(responseData);

      setState(() {
        emotionResult = data["emotion"].toString();
      });
    } catch (e) {
      setState(() {
        emotionResult = "Error: $e";
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Emotion Detection"),
      ),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            TextField(
              controller: _controller,
              decoration: const InputDecoration(labelText: "Enter text"),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: isLoading ? null : predictText,
              child: const Text("Detect Text Emotion"),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: isLoading
                  ? null
                  : () => isRecording ? stopRecording() : startRecording(),
              child: Text(isRecording ? "Stop Recording" : "Start Recording"),
            ),
            const SizedBox(height: 40),

            if (isLoading) ...[
              const CircularProgressIndicator(),
              const SizedBox(height: 20),
            ],

            const Text("Emotion:", style: TextStyle(fontSize: 20)),
            Text(
              emotionResult,
              style: const TextStyle(
                fontSize: 28,
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        ),
      ),
    );
  }
}