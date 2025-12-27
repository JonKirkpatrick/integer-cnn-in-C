#include "compute_programs.hpp"
#include <fstream>
#include <sstream>
#include <cassert>

ComputePrograms& ComputePrograms::instance()
{
    static ComputePrograms instance;
    return instance;
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

    GLuint program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);
    glDeleteShader(shader);

    programs.emplace(name, program);
}

GLuint ComputePrograms::get(const std::string& name) const
{
    auto it = programs.find(name);
    assert(it != programs.end());
    return it->second;
}
