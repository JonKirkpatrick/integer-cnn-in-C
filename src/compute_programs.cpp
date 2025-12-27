#include "compute_programs.hpp"
#include <limits>
#include <fstream>
#include <sstream>
#include <cassert>

ComputePrograms& ComputePrograms::instance()
{
    static ComputePrograms instance;
    return instance;
}

ComputePrograms::~ComputePrograms()
{
    for (auto& [_, prog] : programs)
        glDeleteProgram(prog);
}

void ComputePrograms::loadManifest(const std::string& path)
{
    std::ifstream file(path);
    assert(file && "Failed to open compute shader manifest");

    std::string name, shaderPath;
    while (file >> name)
    {
        if (name[0] == '#')
        {
            file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            continue;
        }

        file >> shaderPath;
        load(name, shaderPath);
    }
}

void ComputePrograms::checkShader(GLuint shader, const std::string& name)
{
    GLint ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        GLint len = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
        std::string log(len, '\0');
        glGetShaderInfoLog(shader, len, nullptr, log.data());
        std::cerr << "Shader [" << name << "] compilation failed:\n" << log << std::endl;
        assert(false);
    }
}

void ComputePrograms::checkProgram(GLuint program, const std::string& name)
{
    GLint ok = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        GLint len = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &len);
        std::string log(len, '\0');
        glGetProgramInfoLog(program, len, nullptr, log.data());
        std::cerr << "Shader [" << name << "] program linking failed:\n" << log << std::endl;
        assert(false);
    }
}

void ComputePrograms::load(const std::string& name, const std::string& path)
{
    std::ifstream file(path);
    assert(file && "Failed to open compute shader");

    std::stringstream ss;
    ss << file.rdbuf();
    std::string source = ss.str();

    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    const char* src = source.c_str();
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    checkShader(shader, name);

    GLuint program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);
    checkProgram(program, name);
    glDetachShader(program, shader);
    glDeleteShader(shader);
    assert(programs.find(name) == programs.end() && "Compute program already loaded");
    programs.emplace(name, program);
}

GLuint ComputePrograms::get(const std::string& name) const
{
    auto it = programs.find(name);
    assert(it != programs.end());
    return it->second;
}