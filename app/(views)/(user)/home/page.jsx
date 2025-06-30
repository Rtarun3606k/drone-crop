"use client";
import Link from 'next/link';

const widgets = [
	{
		label: 'Upload Batches',
		href: '/upload',
		icon: (
			// Upload icon
			<svg
				className="w-6 h-6 mr-3 group-hover:text-black group-hover:stroke-black"
				fill="none"
				stroke="currentColor"
				strokeWidth="2"
				viewBox="0 0 24 24"
			>
				<path
					strokeLinecap="round"
					strokeLinejoin="round"
					d="M4 17v2a2 2 0 002 2h12a2 2 0 002-2v-2M7 9l5-5 5 5M12 4v12"
				/>
			</svg>
		),
	},
	{
		label: 'View Previous Batches',
		href: '/batches',
		icon: (
			// History/Document icon
			<svg
				className="w-6 h-6 mr-3 group-hover:text-black group-hover:stroke-black"
				fill="none"
				stroke="currentColor"
				strokeWidth="2"
				viewBox="0 0 24 24"
			>
				<path
					strokeLinecap="round"
					strokeLinejoin="round"
					d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"
				/>
			</svg>
		),
	},
	{
		label: 'Order Drone Service',
		href: '/order',
		icon: (
			// Provided drone SVG icon with hover color change
			<svg
				className="w-7 h-7 mr-3 group-hover:stroke-black"
				viewBox="0 0 24 24"
				xmlns="http://www.w3.org/2000/svg"
				fill="none"
			>
				<g>
					<path
						d="M4.84,8.18A3.34,3.34,0,1,1,8.18,4.84"
						stroke="currentColor"
						strokeWidth="1.91"
						fill="none"
					/>
					<path
						d="M8.18,19.16a3.34,3.34,0,1,1-3.34-3.34"
						stroke="currentColor"
						strokeWidth="1.91"
						fill="none"
					/>
					<path
						d="M15.82,4.84a3.34,3.34,0,1,1,3.34,3.34"
						stroke="currentColor"
						strokeWidth="1.91"
						fill="none"
					/>
					<path
						d="M19.16,15.82a3.34,3.34,0,1,1-3.34,3.34"
						stroke="currentColor"
						strokeWidth="1.91"
						fill="none"
					/>
					<line
						x1="19.64"
						y1="19.64"
						x2="14.86"
						y2="14.86"
						stroke="currentColor"
						strokeWidth="1.91"
					/>
					<line
						x1="9.14"
						y1="9.14"
						x2="4.36"
						y2="4.36"
						stroke="currentColor"
						strokeWidth="1.91"
					/>
					<line
						x1="9.14"
						y1="14.86"
						x2="4.36"
						y2="19.64"
						stroke="currentColor"
						strokeWidth="1.91"
					/>
					<line
						x1="19.64"
						y1="4.36"
						x2="14.86"
						y2="9.14"
						stroke="currentColor"
						strokeWidth="1.91"
					/>
					<path
						d="M14.86,9.14v5.72a2.86,2.86,0,1,1-5.72,0V9.14a2.86,2.86,0,1,1,5.72,0Z"
						stroke="currentColor"
						strokeWidth="1.91"
						fill="none"
					/>
				</g>
			</svg>
		),
	},
];

export default function ActionWidgets() {
	return (
		<div className="flex justify-center items-center min-h-screen">
			<div className="w-full max-w-5xl px-2 md:px-0">
				<div className="flex flex-col md:flex-row gap-8 md:gap-10 w-full justify-center items-center">
					{widgets.map(({ label, href, icon }) => (
						<Link key={label} href={href} className="w-full md:w-1/3 flex">
							<button
								className="
              group flex items-center justify-center w-full
              py-8 px-6 md:px-8 rounded-2xl font-bold text-2xl md:text-3xl
              bg-black text-green-400 border-2 border-green-500
              transition-all duration-300
              shadow-xl relative overflow-hidden
              hover:text-black hover:border-transparent
              before:content-[''] before:absolute before:inset-0
              before:bg-gradient-to-r before:from-green-400 before:to-green-700
              before:opacity-0 hover:before:opacity-100
              before:transition-opacity before:duration-300
              z-10
            "
								style={{ position: 'relative' }}
							>
								<span className="relative z-20 flex items-center">
									{icon}
									{label}
								</span>
							</button>
						</Link>
					))}
				</div>
			</div>
		</div>
	);
}
