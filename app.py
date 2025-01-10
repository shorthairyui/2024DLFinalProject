from flask import Flask, request, jsonify, send_file
import os
import subprocess

from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 启用跨域支持

# 保存上传图片的目录
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 脚本输出结果保存的目录
RESULT_FOLDER = "./results"
os.makedirs(RESULT_FOLDER, exist_ok=True)

# 提供前端页面
@app.route("/")
def index():
    return send_file("index.html")  # 确保 index.html 在当前目录下

# 路由：处理图片上传和脚本调用
@app.route("/upload", methods=["POST"])
def upload():
    try:
        # 检查文件
        if "file" not in request.files or "keywords" not in request.form:
            return jsonify({"error": "缺少文件或关键词"}), 400

        file = request.files["file"]
        keywords = request.form["keywords"]  # 获取关键词

        if file.filename == "":
            return jsonify({"error": "文件名为空"}), 400

        # 保存上传文件
        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(input_path)

        # 生成结果文件路径
        result_path = os.path.join(RESULT_FOLDER, f"result_{file.filename}")

        # 调用 Bash 脚本
        script_path = "./process.sh"
        result = subprocess.run(
            ["bash", script_path, input_path, result_path, keywords],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # 检查脚本是否运行成功
        if result.returncode != 0:
            return jsonify({"error": "脚本运行失败", "details": result.stderr}), 500

        # 验证结果文件
        if not os.path.exists(result_path):
            return jsonify({"error": "结果文件未生成"}), 404

        # 返回处理结果
        return send_file(result_path, mimetype="image/png")

    except Exception as e:
        return jsonify({"error": "服务器内部错误", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
