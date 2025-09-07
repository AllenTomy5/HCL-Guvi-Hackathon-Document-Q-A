# 🚀 DocuMind AI - Netlify Deployment Guide

## 📋 Overview

This guide will help you deploy your DocuMind AI Document Q&A application to Netlify. Since your app uses Streamlit and requires LLM services, we've created cloud-compatible versions.

## ⚠️ Important Notes

**Netlify Limitations:**
- Netlify is primarily for static sites and serverless functions
- Streamlit apps require persistent servers
- Local Ollama won't work on Netlify

**Recommended Alternatives:**
1. **Heroku** - Better for Python/Streamlit apps
2. **Railway** - Great for Python apps with databases
3. **Render** - Good for Python web services
4. **Streamlit Cloud** - Specifically designed for Streamlit apps

## 🛠️ Netlify Deployment (Experimental)

### Step 1: Prepare Your Repository

1. **Use the cloud-compatible files:**
   - `app_netlify.py` - Modified app for cloud deployment
   - `llm_interface_cloud.py` - Cloud LLM interface
   - `requirements_netlify.txt` - Updated dependencies

2. **Set up environment variables in Netlify:**
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   HUGGINGFACE_API_KEY=your_huggingface_api_key_here
   ```

### Step 2: Deploy to Netlify

1. **Connect your GitHub repository to Netlify**
2. **Set build settings:**
   - Build command: `chmod +x build.sh && ./build.sh`
   - Publish directory: `.`
   - Python version: `3.11`

3. **Deploy!**

## 🌟 Recommended: Streamlit Cloud Deployment

### Step 1: Prepare for Streamlit Cloud

1. **Rename files:**
   ```bash
   mv app_netlify.py app.py
   mv llm_interface_cloud.py llm_interface.py
   mv requirements_netlify.txt requirements.txt
   ```

2. **Create `secrets.toml` for local testing:**
   ```toml
   OPENAI_API_KEY = "your_api_key_here"
   HUGGINGFACE_API_KEY = "your_api_key_here"
   ```

### Step 2: Deploy to Streamlit Cloud

1. **Go to [share.streamlit.io](https://share.streamlit.io)**
2. **Connect your GitHub repository**
3. **Set environment variables:**
   - `OPENAI_API_KEY`
   - `HUGGINGFACE_API_KEY`
4. **Deploy!**

## 🔧 Environment Variables

### Required for Full Functionality:
- `OPENAI_API_KEY` - For OpenAI GPT models
- `HUGGINGFACE_API_KEY` - For Hugging Face models

### Optional:
- `STREAMLIT_SERVER_PORT` - Port for Streamlit (default: 8501)
- `STREAMLIT_SERVER_ADDRESS` - Server address (default: 0.0.0.0)

## 🧪 Testing Locally

1. **Install dependencies:**
   ```bash
   pip install -r requirements_netlify.txt
   ```

2. **Set environment variables:**
   ```bash
   export OPENAI_API_KEY="your_key_here"
   export HUGGINGFACE_API_KEY="your_key_here"
   ```

3. **Run the app:**
   ```bash
   streamlit run app_netlify.py
   ```

## 📱 Features in Cloud Version

✅ **Working Features:**
- Document upload and processing
- Text chunking and embedding
- Vector store operations
- Question answering with cloud LLMs
- Modern dark mode UI
- Responsive design

⚠️ **Limitations:**
- No local Ollama support
- Requires API keys for full functionality
- File uploads are temporary (session-based)

## 🆘 Troubleshooting

### Common Issues:

1. **"No API keys found"**
   - Add environment variables in your deployment platform
   - App will run in demo mode without keys

2. **"Module not found"**
   - Check `requirements_netlify.txt` includes all dependencies
   - Ensure Python version is 3.11+

3. **"Port already in use"**
   - Set `STREAMLIT_SERVER_PORT` environment variable
   - Use different port number

## 🎯 Next Steps

1. **Choose your deployment platform** (recommend Streamlit Cloud)
2. **Set up API keys** for OpenAI or Hugging Face
3. **Test the deployment** with sample documents
4. **Customize the UI** further if needed

## 📞 Support

If you encounter issues:
1. Check the deployment logs
2. Verify environment variables are set
3. Test locally first
4. Check API key permissions and quotas

---

**Happy Deploying! 🚀**
