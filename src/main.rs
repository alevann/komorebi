#[macro_use]
extern crate imgui;
#[macro_use]
extern crate glium;

mod teapot;
use teapot::*;
mod vector;

use glium::{
    glutin::{
        dpi::LogicalSize,
        event::{Event, VirtualKeyCode, WindowEvent},
        event_loop::{ControlFlow, EventLoop},
        platform::run_return::EventLoopExtRunReturn,
        window::WindowBuilder,
        ContextBuilder,
    },
    index::{NoIndices, PrimitiveType},
    Depth, DepthTest, Display, DrawParameters, Frame, IndexBuffer, Program, Surface, VertexBuffer,
};
use imgui::*;
use imgui_glium_renderer::*;
use imgui_winit_support::*;

fn glium_window(title: &str, size: LogicalSize<f32>) -> (Display, EventLoop<()>) {
    let el = EventLoop::new();
    let cb = ContextBuilder::new().with_depth_buffer(24).with_vsync(true);
    let wb = WindowBuilder::new().with_title(title).with_inner_size(size);
    match Display::new(wb, cb, &el) {
        Ok(ds) => (ds, el),
        Err(e) => panic!("Failed to initialize display: {}", e),
    }
}

fn prepare_imgui() -> (Context, WinitPlatform) {
    let mut imgui = Context::create();

    let platform = WinitPlatform::init(&mut imgui);
    let hidpi_factor = platform.hidpi_factor();
    let font_size = (13.0 * hidpi_factor) as f32;
    imgui.fonts().add_font(&[FontSource::DefaultFontData {
        config: Some(FontConfig {
            size_pixels: font_size,
            ..FontConfig::default()
        }),
    }]);
    imgui.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;

    (imgui, platform)
}

struct System {
    display: Display,
    event_loop: EventLoop<()>,
    platform: WinitPlatform,
    renderer: Renderer,
    imgui: Context,
}

fn build_system(window_title: &str, window_size: LogicalSize<f32>) -> System {
    let (ds, el) = glium_window(window_title, window_size);
    let (mut imgui, pf) = prepare_imgui();
    let renderer = if let Ok(renderer) = Renderer::init(&mut imgui, &ds) {
        renderer
    } else {
        panic!("Failed to initialize renderer")
    };
    System {
        event_loop: el,
        display: ds,
        renderer,
        platform: pf,
        imgui,
    }
}

impl System {
    fn main_loop(self, app: &mut Application) {
        let System {
            mut event_loop,
            mut imgui,
            mut platform,
            mut renderer,
            display,
            ..
        } = self;

        let vertex_shader_src = r#"
            #version 140

            in vec3 position;
            in vec3 normal;

            out vec3 v_normal;
            out vec3 v_position;

            uniform mat4 perspective;
            uniform mat4 view;
            uniform mat4 model;

            void main() {
                mat4 modelview = view * model;
                v_normal = transpose(inverse(mat3(modelview))) * normal;
                gl_Position = perspective * modelview * vec4(position, 1.0);
                v_position = gl_Position.xyz / gl_Position.w;
            }
        "#;

        let fragment_shader_src = r#"
            #version 140

            in vec3 v_normal;
            in vec3 v_position;
            
            out vec4 color;
            
            uniform vec3 u_light;
            
            const vec3 ambient_color = vec3(0.2, 0.2, 0.2);
            const vec3 diffuse_color = vec3(0.6, 0.6, 0.6);
            const vec3 specular_color = vec3(1.0, 1.0, 1.0);
            
            void main() {
                float diffuse = max(dot(normalize(v_normal), normalize(u_light)), 0.0);
            
                vec3 camera_dir = normalize(-v_position);
                vec3 half_direction = normalize(normalize(u_light) + camera_dir);
                float specular = pow(max(dot(half_direction, normalize(v_normal)), 0.0), 16.0);
            
                color = vec4(ambient_color + diffuse * diffuse_color + specular * specular_color, 1.0);
            }
        "#;
        let program = Program::from_source(&display, vertex_shader_src, fragment_shader_src, None)
            .expect("Failed to build shaders");

        let mut run = true;
        while run {
            event_loop.run_return(|event, _, flow| {
                platform.handle_event(imgui.io_mut(), display.gl_window().window(), &event);
                if let Event::WindowEvent { event, .. } = event {
                    match event {
                        WindowEvent::CloseRequested => run = false,
                        WindowEvent::KeyboardInput { input, .. } => match input.virtual_keycode {
                            Some(VirtualKeyCode::Escape) => run = false,
                            _ => (),
                        },
                        _ => (),
                    }
                }
                *flow = ControlFlow::Exit;
            });
            if !run {
                break;
            }

            let io = imgui.io_mut();
            platform
                .prepare_frame(io, display.gl_window().window())
                .expect("Failed to prepare frame");
            let mut ui = imgui.frame();
            app.run_ui(&mut ui);
            platform.prepare_render(&ui, display.gl_window().window());

            let mut target = display.draw();
            app.run_gl(&mut target, &program, &display);
            let draw_data = ui.render();
            renderer
                .render(&mut target, draw_data)
                .expect("Rendering failed");
            target.finish().expect("Failed to swap buffers");
        }
    }
}

struct Application {
    background_color: [f32; 4],
    model: Model,
    camera: [[f32; 3]; 3],
    orientation: [f32; 3],
    angle: f32,
}

impl Application {
    pub fn run_ui(&mut self, imgui: &mut Ui) {
        imgui::Window::new(im_str!("Application Settings"))
            .size([300.0, 110.0], Condition::FirstUseEver)
            .build(&imgui, || {
                let mut color = self.background_color;
                let ce = ColorEdit::new(im_str!("Background color"), &mut color);
                if ce.build(&imgui) {
                    self.background_color = color;
                }
                imgui.separator();

                imgui.text(im_str!("Model matrix"));
                let width = imgui.push_item_width(imgui.current_column_width() / 5.0);
                for r in 0..4 {
                    for c in 0..4 {
                        let mut value = self.model.matrix[r][c];
                        let label = im_str!("##model-matrix-{}x{}", r, c);
                        let text = im_str!("{}x{}: %.3f", r, c);
                        let inf = Drag::<f32>::new(&label).display_format(&text).speed(0.1);
                        if inf.build(imgui, &mut value) {
                            self.model.matrix[r][c] = value;
                        }
                        imgui.same_line_with_spacing(0.0, -1.0);
                    }
                    imgui.new_line();
                }
                width.pop(imgui);

                imgui.text(im_str!("Rotation"));
                let width = imgui.push_item_width(imgui.current_column_width() / 4.5);
                Drag::<f32>::new(im_str!("##orx"))
                    .display_format(im_str!("X: %.3f"))
                    .speed(0.05)
                    .build(imgui, &mut self.orientation[0]);
                imgui.same_line(0.0);
                Drag::<f32>::new(im_str!("##ory"))
                    .display_format(im_str!("Y: %.3f"))
                    .speed(0.05)
                    .build(imgui, &mut self.orientation[1]);
                imgui.same_line(0.0);
                Drag::<f32>::new(im_str!("##orz"))
                    .display_format(im_str!("Z: %.3f"))
                    .speed(0.05)
                    .build(imgui, &mut self.orientation[2]);
                imgui.same_line(0.0);
                Drag::<f32>::new(im_str!("##orw"))
                    .display_format(im_str!("Angle: %.3f"))
                    .speed(0.05)
                    .build(imgui, &mut self.angle);
                if imgui.small_button(im_str!("Apply")) {
                    self.model.matrix =
                        rotate_slow(&mut self.model.matrix, self.angle, self.orientation);
                    println!("Rotated!")
                }
                width.pop(imgui);

                imgui.text(im_str!("Camera matrix"));
                let width = imgui.push_item_width(imgui.current_column_width() / 4.0);
                let row_title = ["Position ", "Direction"];
                for r in 0..2 {
                    for c in 0..3 {
                        let mut value = self.camera[r][c];
                        let label = im_str!("##camera-matrix-{}x{}", r, c);
                        let text = im_str!("{}x{}: %.3f", r, c);
                        let inf = Drag::<f32>::new(&label).display_format(&text).speed(0.05);
                        if inf.build(imgui, &mut value) {
                            self.camera[r][c] = value;
                        }
                        imgui.same_line(0.0);
                    }
                    imgui.text(im_str!("{}", row_title[r]));
                }
                width.pop(imgui);
            })
    }
    pub fn run_gl(&mut self, frame: &mut Frame, program: &Program, display: &Display) {
        frame.clear_color_and_depth(
            (
                self.background_color[0],
                self.background_color[1],
                self.background_color[2],
                self.background_color[3],
            ),
            1.0,
        );

        if let None = self.model.vertex_buffer {
            self.model.vertex_buffer = Some(
                VertexBuffer::new(display, &self.model.vertex).expect("Failed to buffer vertices"),
            );
        }
        if let None = self.model.normal_buffer {
            self.model.normal_buffer = Some(
                VertexBuffer::new(display, &self.model.normal).expect("Failed to buffer normals"),
            );
        }
        if let None = self.model.index_buffer {
            self.model.index_buffer = Some(
                IndexBuffer::new(display, PrimitiveType::TrianglesList, &teapot::INDICES)
                    .expect("Failed to build index buffer"),
            );
        }

        let view = view_matrix(&self.camera[0], &self.camera[1], &self.camera[2]);
        let projection = {
            let (width, height) = frame.get_dimensions();
            let aspect_ratio = height as f32 / width as f32;
            let fov: f32 = 3.141592 / 3.0;
            let zfar = 1024.0;
            let znear = 0.1;
            let f = 1.0 / (fov / 2.0).tan();
            [
                [f * aspect_ratio, 0.0, 0.0, 0.0],
                [0.0, f, 0.0, 0.0],
                [0.0, 0.0, (zfar + znear) / (zfar - znear), 1.0],
                [0.0, 0.0, -(2.0 * zfar * znear) / (zfar - znear), 0.0],
            ]
        };
        let light = [-1.0, 0.4, 0.9f32];
        let params = DrawParameters {
            depth: Depth {
                test: DepthTest::IfLess,
                write: true,
                ..Default::default()
            },
            ..Default::default()
        };
        frame.draw(
            (self.model.vertex_buffer.as_ref().unwrap(), self.model.normal_buffer.as_ref().unwrap()),
            self.model.index_buffer.as_ref().unwrap(),
            &program,
            &uniform! { model: self.model.matrix, view: view, perspective: projection, u_light: light },
            &params
        ).unwrap();
    }
}
fn view_matrix(position: &[f32; 3], direction: &[f32; 3], up: &[f32; 3]) -> [[f32; 4]; 4] {
    let f = {
        let f = direction;
        let len = f[0] * f[0] + f[1] * f[1] + f[2] * f[2];
        let len = len.sqrt();
        [f[0] / len, f[1] / len, f[2] / len]
    };

    let s = [
        up[1] * f[2] - up[2] * f[1],
        up[2] * f[0] - up[0] * f[2],
        up[0] * f[1] - up[1] * f[0],
    ];

    let s_norm = {
        let len = s[0] * s[0] + s[1] * s[1] + s[2] * s[2];
        let len = len.sqrt();
        [s[0] / len, s[1] / len, s[2] / len]
    };

    let u = [
        f[1] * s_norm[2] - f[2] * s_norm[1],
        f[2] * s_norm[0] - f[0] * s_norm[2],
        f[0] * s_norm[1] - f[1] * s_norm[0],
    ];

    let p = [
        -position[0] * s_norm[0] - position[1] * s_norm[1] - position[2] * s_norm[2],
        -position[0] * u[0] - position[1] * u[1] - position[2] * u[2],
        -position[0] * f[0] - position[1] * f[1] - position[2] * f[2],
    ];

    [
        [s_norm[0], u[0], f[0], 0.0],
        [s_norm[1], u[1], f[1], 0.0],
        [s_norm[2], u[2], f[2], 0.0],
        [p[0], p[1], p[2], 1.0],
    ]
}
fn rotate_slow(matrix: &mut [[f32; 4]; 4], angle: f32, rotation: [f32; 3]) -> [[f32; 4]; 4] {
    let a = angle;
    let c = a.cos();
    let s = a.sin();

    let axis = rotation[0] * rotation[0] + rotation[1] * rotation[1] + rotation[2] * rotation[2];
    let axis = axis.sqrt();
    let axis = [rotation[0] / axis, rotation[1] / axis, rotation[2] / axis];

    let mut m: [[f32; 4]; 4] = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ];

    m[0][0] = c + (1.0 - c) * axis[0] * axis[0];
    m[0][1] = (1.0 - c) * axis[0] * axis[1] + s * axis[2];
    m[0][2] = (1.0 - c) * axis[0] * axis[2] - s * axis[1];
    m[0][3] = 0.0;

    m[1][0] = (1.0 - c) * axis[1] * axis[0] - s * axis[2];
    m[1][1] = c + (1.0 - c) * axis[1] * axis[1];
    m[1][2] = (1.0 - c) * axis[1] * axis[2] + s * axis[0];
    m[1][3] = 0.0;

    m[2][0] = (1.0 - c) * axis[2] * axis[0] + s * axis[1];
    m[2][1] = (1.0 - c) * axis[2] * axis[1] - s * axis[0];
    m[2][2] = c + (1.0 - c) * axis[2] * axis[2];
    m[2][3] = 0.0;

    m[3][0] = 0.0;
    m[3][1] = 0.0;
    m[3][2] = 0.0;
    m[3][3] = 1.0;

    mult_m4(m, *matrix)
}
fn mult_m4(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ];

    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    out
}

impl Default for Application {
    fn default() -> Self {
        Application {
            background_color: [0.0, 0.0, 0.0, 1.0],
            model: Model::new(),
            camera: [[2.0, 1.6, 1.0], [-2.0, -1.0, 1.0], [0.0, 1.0, 0.0]],
            orientation: [0.0, 0.0, 0.0],
            angle: 0.0,
        }
    }
}

/*#[derive(Copy, Clone)]
struct Vertex {
    position: [f32; 3]
}
implement_vertex!(Vertex, position);

#[derive(Copy, Clone)]
struct Normal {
    normal: [f32; 3]
}
implement_vertex!(Normal, normal);*/

struct Model {
    vertex: Vec<Vertex>,
    normal: Vec<Normal>,
    matrix: [[f32; 4]; 4],
    vertex_buffer: Option<VertexBuffer<Vertex>>,
    normal_buffer: Option<VertexBuffer<Normal>>,
    index_buffer: Option<IndexBuffer<u16>>,
}

impl Model {
    fn new() -> Model {
        let matrix = [
            [0.01, 0.0, 0.0, 0.0],
            [0.0, 0.01, 0.0, 0.0],
            [0.0, 0.0, 0.01, 0.0],
            [0.0, 0.0, 2.0, 1.0f32],
        ];
        let vertices = teapot::VERTICES.to_vec();
        let normal = teapot::NORMALS.to_vec();

        /*let (models, materials) = tobj::load_obj(
            "./assets/tree/obj/tree.obj",
            &tobj::LoadOptions {
                single_index: true,
                ..tobj::LoadOptions::default()
            }
        ).expect("Failed to load OBJ file");
        println!("Number of models = {}", models.len());
        for (i, m) in models.iter().enumerate() {
            let mesh = &m.mesh;
            println!("Mesh \'{}\' has {} vertexes and {} normals", m.name, mesh.positions.len(), mesh.normals.len());
            let mut next_face = 0;
            for face in 0..mesh.face_arities.len() {
                let end = next_face + mesh.face_arities[face] as usize;
                let face_indices = &mesh.indices[next_face..end];
                println!("Face[{}] indices = {:?}", face, face_indices);
                next_face = end;
            }
            for vtx in 0..mesh.positions.len() / 3 {
                vertices.push(Vertex {
                    position: [
                        mesh.positions[3 * vtx],
                        mesh.positions[3 * vtx + 1],
                        mesh.positions[3 * vtx + 2]
                    ]
                });
            }
            for nrm in 0..mesh.normals.len() / 3 {
                normal.push(Normal {
                    normal: [
                        mesh.normals[3 * nrm],
                        mesh.normals[3 * nrm + 1],
                        mesh.normals[3 * nrm + 2]
                    ]
                });
            }
        }
        println!("Model vertex count: {}\nModel normal count: {}", vertices.len(), normal.len());*/
        Model {
            vertex: vertices,
            matrix,
            vertex_buffer: None,
            normal,
            normal_buffer: None,
            index_buffer: None,
        }
    }
}

fn main() {
    let system = build_system(
        "komorebi",
        LogicalSize {
            width: 1024.0,
            height: 768.0,
        },
    );
    system.main_loop(&mut Default::default());
}
