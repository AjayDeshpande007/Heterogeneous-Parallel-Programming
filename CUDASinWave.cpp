#include "vmath.h"
#include<GL/glew.h>
#include<Windows.h>
#include<stdio.h>

#include<GL/GL.h>
#include<cuda_gl_interop.h>
#include<cuda_runtime.h>

#pragma comment(lib,"glew32.lib")
#pragma comment(lib,"opengl32.lib")
#pragma comment(lib,"cudart.lib")

#define WIN_WIDTH 800
#define WIN_HEIGHT 600
const int g_mesh_width = 2048;
const int g_mesh_height = 2048;

#define MYARRAYSIZE g_mesh_width * g_mesh_height * 4
using namespace vmath;

enum
{
	AMC_ATTRIBUTE_POSITION = 0,
	AMC_ATTRIBUTE_COLOR,
	AMC_ATTRIBUTE_NORMAL,
	AMC_ATTRIBUTE_TEXCOORD
};

float pos[g_mesh_width][g_mesh_height][4];

struct cudaGraphicsResource* graphicsResource = NULL;
GLuint vbo_GPU;

float animationTime = 0.0f;
BOOL onGPU = FALSE;
cudaError_t error;

GLuint vao;
GLuint vbo;
GLuint mvpUniform;
mat4 perspectivegraphicProjectionMatrix;

HDC ghdc = NULL;
HWND ghwnd = NULL;
HGLRC ghrc = NULL;
bool gbActiveWindow = false;
FILE* gpFile = NULL;
bool gbFullScreen = TRUE;
DWORD dwStyle;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };
GLuint gVertexShaderObject;
GLuint gFragmentShaderObject;
GLuint gShaderProgramObject;

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
	WNDCLASSEX wndclass;
	HWND hwnd;
	MSG msg;
	int iRet = 0;
	bool bDone = false;

	TCHAR szAppName[] = TEXT("OGL_Programmable Pipeline");

	int initialize(void);
	void display(void);

	if (fopen_s(&gpFile, "CUDA.txt", "w") != 0)
	{
		MessageBox(NULL, TEXT("File not created"), TEXT("Error"), MB_OK | MB_ICONERROR);
		exit(0);
	}
	else
	{
		fprintf(gpFile, "File Created Successfully for CUDA\n");
	}

	wndclass.cbSize = sizeof(WNDCLASSEX);
	wndclass.style = CS_HREDRAW | CS_VREDRAW;
	wndclass.cbClsExtra = 0;
	wndclass.cbWndExtra = 0;
	wndclass.lpfnWndProc = WndProc;
	wndclass.lpszClassName = szAppName;
	wndclass.lpszMenuName = NULL;
	wndclass.hInstance = hInstance;
	wndclass.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
	wndclass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);

	RegisterClassEx(&wndclass);

	hwnd = CreateWindowEx(WS_EX_APPWINDOW, szAppName, TEXT("CUDA"), WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE, 100, 100, WIN_WIDTH, WIN_HEIGHT, NULL, NULL, hInstance, NULL);

	if (hwnd == NULL)
	{
		MessageBox(NULL, TEXT("Window is not created"), TEXT("Error"), MB_OK | MB_ICONERROR);
		exit(0);
	}

	ghwnd = hwnd;

	iRet = initialize();

	if (iRet == -1)
	{
		fprintf(gpFile, "Choose Pixel Format error\n");
		exit(0);
	}
	else if (iRet == -2)
	{
		fprintf(gpFile, "Error in Set pixel format\n");
		exit(0);
	}
	else if (iRet == -3)
	{
		fprintf(gpFile, "wglMakeCurrent Failed");
		exit(0);
	}
	else
	{
		fprintf(gpFile, "initialize function successful\n");
	}
	ShowWindow(hwnd, SW_SHOWMAXIMIZED);
	SetForegroundWindow(hwnd);
	SetFocus(hwnd);

	while (bDone == false)
	{
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			if (msg.message == WM_QUIT)
			{
				bDone = true;
			}
			else
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
		}
		else
		{
			if (gbActiveWindow == true)
			{

			}
			display();
		}
	}
	return((int)msg.wParam);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	void ToggleFullScreen(void);
	void reshape(int, int);
	void uninitialize(void);

	switch (iMsg)
	{

	case WM_CREATE:
		ToggleFullScreen();
		break;
	case WM_SETFOCUS:
		gbActiveWindow = true;
		break;

	case WM_KILLFOCUS:
		gbActiveWindow = false;
		break;

	case WM_KEYDOWN:
		switch (wParam)
		{
		case 'c':
		case 'C':
			onGPU = false;
			break;
		case 'g':
		case 'G':
			onGPU = true;
			break;
		case VK_ESCAPE:
			DestroyWindow(hwnd);

		case 0x46:
			ToggleFullScreen();
			break;
		}
		break;

	case WM_SIZE:
		reshape(LOWORD(lParam), HIWORD(lParam));
		break;

	case WM_CLOSE:
		DestroyWindow(hwnd);
		break;

	case WM_DESTROY:
		uninitialize();
		PostQuitMessage(0);
		break;
	}
	return DefWindowProc(hwnd, iMsg, wParam, lParam);
}

int initialize()
{
	void reshape(int, int);
	void uninitialize(void);
	void VertexShader(void);
	void FragmentShader(void);
	void DrawTriangle(void);
	PIXELFORMATDESCRIPTOR pfd;
	int iPixelFormatIndex;
	GLenum result;
	int devCount;
	if (cudaGetDeviceCount(&devCount) != cudaSuccess)
	{
		fprintf(gpFile, "Could not get CUDA device\n");
		uninitialize();
		exit(0);
	}
	else if (devCount == 0)
	{
		fprintf(gpFile, "\nNo CUDA Device\n");
	}
	else
	{
		fprintf(gpFile, "CUDA device set successful %d\n", devCount);
		cudaSetDevice(0);
	}

	ZeroMemory(&pfd, sizeof(PIXELFORMATDESCRIPTOR));
	pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
	pfd.nVersion = 1;
	pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
	pfd.iPixelType = PFD_TYPE_RGBA;
	pfd.cColorBits = 32;
	pfd.cRedBits = 8;
	pfd.cGreenBits = 8;
	pfd.cBlueBits = 8;
	pfd.cAlphaBits = 8;
	pfd.cDepthBits = 32;

	ghdc = GetDC(ghwnd);

	iPixelFormatIndex = ChoosePixelFormat(ghdc, &pfd);

	if (iPixelFormatIndex == 0)
	{
		return -1;
	}

	if (SetPixelFormat(ghdc, iPixelFormatIndex, &pfd) == FALSE)
	{
		return -2;
	}

	ghrc = wglCreateContext(ghdc);

	if (ghrc == NULL)
	{
		return -3;
	}

	if (wglMakeCurrent(ghdc, ghrc) == NULL)
	{
		return -4;
	}
	result = glewInit();
	if (result != GLEW_OK)
	{
		fprintf(gpFile, "Failed to create glewInit()\n");
		uninitialize();
		DestroyWindow(ghwnd);
	}
	for (int i = 0; i < g_mesh_width; i++)
	{
		for (int j = 0; j < g_mesh_height; j++)
		{
			for (int k = 0; k < 4; k++)
			{
				pos[i][j][k] = 0;
			}
		}
	}//Make array initialize to Zero
	fprintf(gpFile, "gmesh_width %d\n", g_mesh_width);
	fprintf(gpFile, "gmesh_height %d\n", g_mesh_height);



	VertexShader();
	FragmentShader();

	gShaderProgramObject = glCreateProgram();
	// attach vertex shader to shader program
	glAttachShader(gShaderProgramObject, gVertexShaderObject);

	glAttachShader(gShaderProgramObject, gFragmentShaderObject);

	//binding to vertex attribute
	glBindAttribLocation(gShaderProgramObject, AMC_ATTRIBUTE_POSITION, "vPosition");

	// Link the shader program
	glLinkProgram(gShaderProgramObject);
	mvpUniform = glGetUniformLocation(gShaderProgramObject, "u_mvp_matrix");

	DrawTriangle();



	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);


	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClearDepth(1.0f);

	perspectivegraphicProjectionMatrix = mat4::identity();
	reshape(WIN_WIDTH, WIN_HEIGHT);

	return(0);

}



void reshape(int width, int height)
{
	//code added
	if (height == 0)
	{
		height = 1;
	}
	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	perspectivegraphicProjectionMatrix = perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);
}


void display(void)
{
	void launchCPUKernel(unsigned int, unsigned int, float);
	void launchCudaKernel(float4*, unsigned int, unsigned int, float);
	void uninitialize();
	float4* pPos = NULL;
	size_t byteCount;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


	glUseProgram(gShaderProgramObject);
	//Declaration of matrices
	mat4 modelViewMatrix;
	mat4 modelViewProjectMatrix;

	//initialize above matrices to identity
	modelViewMatrix = mat4::identity();
	modelViewProjectMatrix = mat4::identity();

	//Do necessary transformation

	modelViewMatrix = translate(0.0f, 0.0f, -2.0f);

	//Do necessary matrix multiplication
	modelViewProjectMatrix = perspectivegraphicProjectionMatrix * modelViewMatrix;
	//send necessary matrices to shader in respective uniform
	glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectMatrix);
	glBindVertexArray(vao);

	if (onGPU == TRUE)
	{
		glBindBuffer(GL_ARRAY_BUFFER, vbo_GPU);
		//Map with resource
		error = cudaGraphicsMapResources(1, &graphicsResource, 0);
		if (error != cudaSuccess)
		{
			fprintf(gpFile, "Error : in cudaGraphics Resources\n");
			uninitialize();
			exit(0);
		}

		error = cudaGraphicsResourceGetMappedPointer((void**)&pPos, &byteCount, graphicsResource);
		if (error != cudaSuccess)
		{
			fprintf(gpFile, "Error : cudaGraphicsResources\n");
			uninitialize();
			exit(0);
		}
		//fprintf(gpFile, "Byte Count %lf\n", byteCount);

		//cuda krnel launch
		launchCudaKernel(pPos, g_mesh_width, g_mesh_height, animationTime);

		//unmap the resource 
		error = cudaGraphicsUnmapResources(1, &graphicsResource, 0);

		if (error != cudaSuccess)
		{
			fprintf(gpFile, "Error : cudaGraphicsUnmapResources\n");
			uninitialize();
			exit(0);
		}
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
	else
	{
		launchCPUKernel(g_mesh_width, g_mesh_height, animationTime);

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(pos), pos, GL_DYNAMIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	if (onGPU)
	{
		glBindBuffer(GL_ARRAY_BUFFER, vbo_GPU);
	}
	else
	{
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	}
	//Bind with vao
	//(This will avoid repetative steps)
	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);


	//draw neseaary scene
	glDrawArrays(GL_POINTS, 0, g_mesh_width * g_mesh_height);

	glBindVertexArray(0);

	glUseProgram(0);
	animationTime += 0.1f;


	SwapBuffers(ghdc);

}

void launchCPUKernel(unsigned int width, unsigned int height, float time)
{
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			float u = i / (float)width;
			float v = j / (float)height;

			u = u * 2.0f - 1.0f;
			v = v * 2.0f - 1.0f;

			float frequency = 4.0f;
			float w = sinf(frequency * u + time) * cosf(frequency * v + time) * 0.5f;

			for (int k = 0; k < 4; k++)
			{

				if (k == 0)
					pos[i][j][k] = u;
				if (k == 1)
					pos[i][j][k] = w;
				if (k == 2)
					pos[i][j][k] = v;
				if (k == 3)
					pos[i][j][k] = 1.0f;

			}
		}
	}
}

void ToggleFullScreen()
{
	MONITORINFO mi;
	if (gbFullScreen == false)
	{
		dwStyle = GetWindowLong(ghwnd, GWL_STYLE);

		if (dwStyle & WS_OVERLAPPEDWINDOW)
		{
			mi = { sizeof(MONITORINFO) };

			if (GetWindowPlacement(ghwnd, &wpPrev) && GetMonitorInfo(MonitorFromWindow(ghwnd, MONITORINFOF_PRIMARY), &mi))
			{
				SetWindowLong(ghwnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);
				SetWindowPos(ghwnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left, mi.rcMonitor.bottom - mi.rcMonitor.top, SWP_NOZORDER | SWP_FRAMECHANGED);

			}
		}

		ShowCursor(false);
		gbFullScreen = true;
	}
	else
	{
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOZORDER | SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER);
		ShowCursor(false);
		gbFullScreen = false;
	}


}


void VertexShader(void)
{
	GLint iShaderCompileStatus;
	GLint iInfoLogLength;
	GLchar* szInfoLog;
	void uninitialize(void);

	gVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);
	const GLchar* vertexShaderSourceCode = "#version 450 core" \
		"\n" \
		"in vec4 vPosition;" \
		"uniform mat4 u_mvp_matrix;" \
		"void main(void)" \
		"{" \
		"gl_Position = u_mvp_matrix * vPosition;"
		"}";

	glShaderSource(gVertexShaderObject, 1, (const GLchar**)&vertexShaderSourceCode, NULL);
	glCompileShader(gVertexShaderObject);
	iShaderCompileStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;

	glGetShaderiv(gVertexShaderObject, GL_COMPILE_STATUS, &iShaderCompileStatus);

	if (iShaderCompileStatus == GL_FALSE)
	{
		glGetShaderiv(gVertexShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (GLchar*)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(gVertexShaderObject, iInfoLogLength, &written, szInfoLog);
				fprintf_s(gpFile, "Failed in Vertex Shader Object");
				free(szInfoLog);
				uninitialize();
				DestroyWindow(ghwnd);
				exit(0);
			}
		}
	}
}

void FragmentShader(void)
{
	void uninitialize(void);
	GLint iShaderCompileStatus;
	GLint iInfoLogLength;
	GLchar* szInfoLog;
	gFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);
	// write fragment shader code
	const GLchar* FragmentShaderSourceCode = "#version 450 core" \
		"\n" \
		"out vec4 FragColor;" \
		"void main(void)" \
		"{" \
		"FragColor = vec4(1.0,1.0,0.0,1.0);" \
		"}";
	glShaderSource(gFragmentShaderObject, 1, (const GLchar**)&FragmentShaderSourceCode, NULL);
	glCompileShader(gFragmentShaderObject);
	iShaderCompileStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;
	glGetShaderiv(gFragmentShaderObject, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if (iShaderCompileStatus == GL_FALSE)
	{
		glGetShaderiv(gFragmentShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		{
			szInfoLog = (GLchar*)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(gFragmentShaderObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Fragment Shader compile Log : \n%s.\n", szInfoLog);
				free(szInfoLog);
				DestroyWindow(ghwnd);
				uninitialize();
				exit(0);
			}
		}
	}
}

void DrawTriangle(void)
{
	void uninitialize();


	GLint iInfoLogLength;
	GLchar* szInfoLog;
	GLint iProgramLinkStatus;


	// Linking time error checking of shader program
	iProgramLinkStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;
	// error checking step [1] 
	glGetProgramiv(gShaderProgramObject, GL_LINK_STATUS, &iProgramLinkStatus);
	// error checking step [2] 
	if (iProgramLinkStatus == GL_FALSE)
	{
		// error checking step [3] 
		glGetProgramiv(gShaderProgramObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		// error checking step [4] 
		if (iInfoLogLength > 0)
		{
			szInfoLog = (GLchar*)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetProgramInfoLog(gShaderProgramObject,
					iInfoLogLength,
					&written,
					szInfoLog);
				// error checking step [5] 
				fprintf(gpFile, "Shader Program Link Log : \n%s.\n", szInfoLog);
				free(szInfoLog);
				DestroyWindow(ghwnd);
				exit(0);
			}
		}
	}

	//Create vao
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * MYARRAYSIZE, NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);




	//GPU vbo
	glGenBuffers(1, &vbo_GPU);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_GPU);

	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * MYARRAYSIZE, NULL, GL_DYNAMIC_DRAW);


	//glBindBuffer(GL_ARRAY_BUFFER, 0);


	//CUDA - Buffer Registered in CUDA (GPU)
	error = cudaGraphicsGLRegisterBuffer(&graphicsResource, vbo_GPU, cudaGraphicsMapFlagsWriteDiscard);
	if (error != cudaSuccess)
	{

		fprintf(gpFile, "Error in cudaGraphicsGLRegisterBuffer \n%s\n", cudaGetErrorString(error));
		uninitialize();
		exit(0);
	}

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

}

void uninitialize(void)
{
	if (vbo)
	{
		glDeleteBuffers(1, &vbo);
		vbo = 0;
	}
	if (vbo_GPU)
	{
		glDeleteBuffers(1, &vbo_GPU);
		vbo_GPU = 0;
	}
	if (vao)
	{
		glDeleteVertexArrays(1, &vao);
		vao = 0;
	}

	if (gShaderProgramObject)
	{
		GLsizei shaderCount;
		GLsizei shaderNumber;
		glUseProgram(gShaderProgramObject);

		//ask program how many ShaderObject
		glGetProgramiv(gShaderProgramObject, GL_ATTACHED_SHADERS, &shaderCount);

		GLuint* pShaders = (GLuint*)malloc(sizeof(GLuint) * shaderCount);

		if (pShaders)
		{
			glGetAttachedShaders(gShaderProgramObject, shaderCount, &shaderCount, pShaders);

			for (shaderNumber = 0; shaderNumber < shaderCount; shaderNumber++)
			{
				//Detach one by one shader
				glDetachShader(gShaderProgramObject, pShaders[shaderNumber]);

				//Delete the detached shader
				glDeleteShader(pShaders[shaderNumber]);

				pShaders[shaderNumber] = 0;
			}
			free(pShaders);
		}
		glDeleteProgram(gShaderProgramObject);
		gShaderProgramObject = 0;
		glUseProgram(0);
	}

	if (gbFullScreen == true)
	{
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd, &wpPrev);

		SetWindowPos(ghwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOZORDER | SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER);

		ShowCursor(true);
	}

	if (wglGetCurrentContext() == ghrc)
	{
		wglMakeCurrent(NULL, NULL);
	}

	if (ghrc)
	{
		wglDeleteContext(ghrc);
		ghrc = NULL;
	}

	if (ghdc)
	{
		ReleaseDC(ghwnd, ghdc);
		ghdc = NULL;
	}

	error = cudaGraphicsUnregisterResource(graphicsResource);
	if (error != cudaSuccess)
	{
		fprintf(gpFile, "Error : Cuda Graphics is unregisteres\n");

		exit(0);
	}

	if (gpFile)
	{
		fprintf(gpFile, "File Closed Successfully\n");
		fclose(gpFile);
		gpFile = NULL;
	}

}
